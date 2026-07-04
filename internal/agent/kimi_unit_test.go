package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKimiName(t *testing.T) {
	t.Parallel()

	a := NewKimiAgent("")
	assert.Equal(t, "kimi", a.Name())
	assert.Equal(t, "kimi", a.CommandName())
}

func TestKimiCommandLine(t *testing.T) {
	t.Parallel()

	a := NewKimiAgent("kimi").WithModel("kimi-k2.7-code").WithAgentic(true).(*KimiAgent)
	cl := a.CommandLine()
	assert.Equal(t, "kimi "+strings.Join(a.buildArgs("<prompt-file>"), " "), cl)
	assert.Contains(t, cl, "--output-format stream-json")
	assert.Contains(t, cl, "--model")
	assert.NotContains(t, cl, "--yolo")
}

func TestKimiBuildArgs(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		model       string
		agentic     bool
		sessionID   string
		wantArgs    []string
		excludeArgs []string
	}{
		{
			name:     "review mode defaults",
			wantArgs: []string{"-p", "@/tmp/prompt.txt", "--output-format", "stream-json"},
			excludeArgs: []string{
				"--model", "--yolo", "-S",
			},
		},
		{
			name:        "agentic mode is ignored; --yolo not emitted",
			agentic:     true,
			excludeArgs: []string{"--yolo", "--model"},
		},
		{
			name:      "model flag included",
			model:     "kimi-k2.7-code",
			wantArgs:  []string{"--model", "kimi-k2.7-code"},
			excludeArgs: []string{"--yolo"},
		},
		{
			name:      "session flag included",
			sessionID: "ses_123",
			wantArgs:  []string{"-S", "ses_123"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			a := NewKimiAgent("kimi")
			a.Model = tt.model
			a.Agentic = tt.agentic
			a.SessionID = tt.sessionID

			args := a.buildArgs("/tmp/prompt.txt")
			for _, want := range tt.wantArgs {
				assert.Contains(t, args, want)
			}
			for _, exclude := range tt.excludeArgs {
				assert.NotContains(t, args, exclude)
			}
		})
	}
}

func TestKimiRejectInvalidResumeSession(t *testing.T) {
	t.Parallel()

	a := NewKimiAgent("kimi").WithSessionID("-bad-session").(*KimiAgent)
	args := a.buildArgs("/tmp/prompt.txt")
	assert.NotContains(t, args, "-bad-session")
	assert.NotContains(t, args, "-S")
}

func TestKimiWithChaining(t *testing.T) {
	t.Parallel()

	a := NewKimiAgent("kimi")
	b := a.WithModel("m1").WithReasoning(ReasoningThorough).WithAgentic(true).(*KimiAgent)
	assert.Equal(t, "m1", b.Model)
	assert.Equal(t, ReasoningThorough, b.Reasoning)
	assert.True(t, b.Agentic)
	assert.Equal(t, "kimi", b.Command)
}

func TestKimiModelFlag(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name         string
		model        string
		wantModel    bool
		wantContains string
	}{
		{name: "no model omits flag", model: ""},
		{
			name:         "explicit model includes flag",
			model:        "kimi-k2.7-code",
			wantModel:    true,
			wantContains: "kimi-k2.7-code",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			a := NewKimiAgent("kimi")
			a.Model = tt.model
			cl := a.CommandLine()
			assert.Contains(t, cl, "--output-format stream-json")
			if tt.wantModel {
				assert.Contains(t, cl, "--model")
				assert.Contains(t, cl, tt.wantContains)
			} else {
				assert.NotContains(t, cl, "--model")
			}
		})
	}
}

func TestKimiAgenticFlag(t *testing.T) {
	t.Parallel()

	a := NewKimiAgent("kimi")
	assert.NotContains(t, a.CommandLine(), "--yolo")

	agentic := a.WithAgentic(true).(*KimiAgent)
	assert.NotContains(t, agentic.CommandLine(), "--yolo")
}

func TestKimiSessionFlag(t *testing.T) {
	t.Parallel()

	a := NewKimiAgent("kimi").WithSessionID("ses_123").(*KimiAgent)
	cl := a.CommandLine()
	assert.Contains(t, cl, "-S")
	assert.Contains(t, cl, "ses_123")
}

func TestKimiBuildArgsUsesPromptFileReference(t *testing.T) {
	t.Parallel()

	a := NewKimiAgent("kimi")
	args := a.buildArgs("/tmp/roborev-kimi-prompt.txt")
	assert.Contains(t, args, "-p")
	assert.Contains(t, args, "@/tmp/roborev-kimi-prompt.txt")
}

func TestParseKimiJSON(t *testing.T) {
	t.Parallel()

	lines := strings.Join([]string{
		unitMakeKimiEvent("assistant", "First thought."),
		unitMakeKimiEvent("assistant", "Final review."),
		unitMakeKimiEvent("meta", "session hint"),
	}, "\n") + "\n"

	var outputBuf strings.Builder
	result, err := parseKimiJSON(
		strings.NewReader(lines), newSyncWriter(&outputBuf),
	)
	require.NoError(t, err)
	assert.Equal(t, "Final review.", result)
	assert.Equal(t, 3, strings.Count(outputBuf.String(), "\n"))
}

func TestParseKimiJSON_IgnoresMeta(t *testing.T) {
	t.Parallel()

	lines := strings.Join([]string{
		unitMakeKimiMetaEvent("session.resume_hint", "session-id", "kimi -r session-id"),
		unitMakeKimiEvent("assistant", "Review content."),
	}, "\n") + "\n"

	result, err := parseKimiJSON(strings.NewReader(lines), nil)
	require.NoError(t, err)
	assert.Equal(t, "Review content.", result)
	assert.NotContains(t, result, "session-id")
}

func TestParseKimiJSON_SanitizesControlChars(t *testing.T) {
	t.Parallel()

	lines := unitMakeKimiEvent("assistant", "\x1b[31mred\x1b[0m and \x1b]0;evil\x07safe") + "\n"
	result, err := parseKimiJSON(strings.NewReader(lines), nil)
	require.NoError(t, err)
	assert.NotContains(t, result, "\x1b")
	assert.NotContains(t, result, "\x07")
	assert.Contains(t, result, "red")
	assert.Contains(t, result, "safe")
	assert.NotContains(t, result, "evil")
}

func TestParseKimiJSON_ReadError(t *testing.T) {
	t.Parallel()

	result, err := parseKimiJSON(
		&unitFailAfterReader{data: unitMakeKimiEvent("assistant", "partial") + "\n"},
		nil,
	)
	require.Error(t, err)
	assert.Contains(t, result, "partial")
}

func TestParseKimiJSON_NoValidEvents(t *testing.T) {
	t.Parallel()

	lines := `{"unknown":"event"}
not json at all
`
	_, err := parseKimiJSON(strings.NewReader(lines), nil)
	require.Error(t, err)
	assert.ErrorIs(t, err, errNoKimiJSON)
}

func TestParseKimiJSON_ResetsAfterToolUse(t *testing.T) {
	t.Parallel()

	lines := strings.Join([]string{
		unitMakeKimiEvent("assistant", "I will read the files first."),
		`{"role":"tool","tool_call_id":"call_1","content":"file contents"}`,
		unitMakeKimiEvent("assistant", "Final review finding."),
	}, "\n") + "\n"

	result, err := parseKimiJSON(strings.NewReader(lines), nil)
	require.NoError(t, err)
	assert.Equal(t, "Final review finding.", result)
	assert.NotContains(t, result, "read the files")
}

func unitMakeKimiEvent(role, content string) string {
	ev := map[string]any{"role": role, "content": content}
	b, err := json.Marshal(ev)
	if err != nil {
		panic("unitMakeKimiEvent: " + err.Error())
	}
	return string(b)
}

func unitMakeKimiMetaEvent(metaType, sessionID, command string) string {
	ev := map[string]any{
		"role":       "meta",
		"type":       metaType,
		"session_id": sessionID,
		"command":    command,
		"content":    "To resume this session: " + command,
	}
	b, err := json.Marshal(ev)
	if err != nil {
		panic("unitMakeKimiMetaEvent: " + err.Error())
	}
	return string(b)
}
