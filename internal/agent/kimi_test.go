//go:build integration

package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKimiReviewModelFlag(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	tests := []struct {
		name      string
		model     string
		wantFlag  bool
		wantModel string
	}{
		{
			name:     "no model omits --model from args",
			model:    "",
			wantFlag: false,
		},
		{
			name:      "explicit model passes --model to subprocess",
			model:     "kimi-k2.7-code",
			wantFlag:  true,
			wantModel: "kimi-k2.7-code",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, args, _ := runMockKimiReview(
				t, tt.model, "review this", nil,
			)
			args = strings.TrimSpace(args)

			assertContains(t, args, "--output-format stream-json")
			if tt.wantFlag {
				assertContains(t, args, "--model")
				assertContains(t, args, tt.wantModel)
			} else {
				assertNotContains(t, args, "--model")
			}
		})
	}
}

func TestKimiReviewPassesPromptViaFileReference(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	prompt := "Review this commit carefully"
	mock := mockAgentCLI(t, MockCLIOpts{
		CaptureArgs:       true,
		CapturePromptFile: true,
		StdoutLines:       []string{makeKimiEvent("assistant", "ok")},
	})

	a := NewKimiAgent(mock.CmdPath)
	_, err := a.Review(context.Background(), t.TempDir(), "HEAD", prompt, nil)
	require.NoError(t, err)

	args := readMockArgs(t, mock.ArgsFile)
	var promptRef string
	for i, arg := range args {
		if arg == "-p" && i+1 < len(args) {
			promptRef = args[i+1]
			break
		}
	}
	require.NotEmpty(t, promptRef, "expected -p argument")
	require.True(t, strings.HasPrefix(promptRef, "@"), "expected prompt file reference, got %q", promptRef)

	content, err := os.ReadFile(mock.PromptFile)
	require.NoError(t, err)
	assert.Equal(t, prompt, strings.TrimSpace(string(content)))
}

func TestKimiReviewArgsContainOutputFormat(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	_, args, _ := runMockKimiReview(t, "", "prompt", nil)
	assertContains(t, strings.TrimSpace(args), "--output-format stream-json")
}

func TestKimiReviewAgenticIncludesYolo(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	mock := mockAgentCLI(t, MockCLIOpts{
		CaptureArgs: true,
		StdoutLines: []string{makeKimiEvent("assistant", "ok")},
	})

	a := NewKimiAgent(mock.CmdPath).WithAgentic(true).(*KimiAgent)
	_, err := a.Review(context.Background(), t.TempDir(), "HEAD", "prompt", nil)
	require.NoError(t, err)

	args := readMockArgs(t, mock.ArgsFile)
	assert.Contains(t, args, "--yolo")
}

func TestKimiReviewParsesJSONStream(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	stdoutLines := []string{
		makeKimiEvent("assistant", "First thought."),
		makeKimiEvent("assistant", "Final review."),
		makeKimiMetaEvent("session.resume_hint", "session-id", "kimi -r session-id"),
	}

	result, _, _ := runMockKimiReview(t, "", "prompt", stdoutLines)
	assertEqual(t, "Final review.", result)
}

func TestKimiReviewStreamsToOutput(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	stdoutLines := []string{
		makeKimiEvent("assistant", "Hello world"),
	}

	var outputBuf bytes.Buffer
	result, _, _, err := executeKimiReviewTest(t, reviewTestOpts{
		MockOpts: MockCLIOpts{
			CaptureArgs: true,
			StdoutLines: stdoutLines,
		},
		Prompt: "prompt",
		Writer: &outputBuf,
	})
	require.NoError(t, err, "Review failed: %v")

	assertEqual(t, result, "Hello world")
	outStr := outputBuf.String()
	assertContains(t, outStr, `"role":"assistant"`)
}

func TestKimiReviewPartialOnError(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	stdoutLines := []string{
		makeKimiEvent("assistant", "Partial review text"),
	}

	_, _, _, err := executeKimiReviewTest(t, reviewTestOpts{
		MockOpts: MockCLIOpts{
			CaptureArgs: true,
			StdoutLines: stdoutLines,
			ExitCode:    1,
		},
		Prompt: "prompt",
	})
	require.Error(t, err)
	assertContains(t, err.Error(), "Partial review text")
}

func TestKimiReviewNoOutput(t *testing.T) {
	t.Parallel()
	skipIfWindows(t)

	stdoutLines := []string{
		makeKimiMetaEvent("session.resume_hint", "session-id", "kimi -r session-id"),
	}

	result, _, _ := runMockKimiReview(t, "", "prompt", stdoutLines)
	assertEqual(t, "No review output generated", result)
}

func executeKimiReviewTest(t *testing.T, opts reviewTestOpts) (string, string, string, error) {
	t.Helper()

	if opts.Prompt == "" {
		require.NotEmpty(t, opts.Prompt, "executeKimiReviewTest requires an explicit Prompt")
	}

	mock := mockAgentCLI(t, opts.MockOpts)

	a := NewKimiAgent(mock.CmdPath)
	if opts.Model != "" {
		a.Model = opts.Model
	}
	if opts.SessionID != "" {
		a = a.WithSessionID(opts.SessionID).(*KimiAgent)
	}

	out, err := a.Review(
		context.Background(), t.TempDir(),
		"HEAD", opts.Prompt, opts.Writer,
	)

	var argsBytes []byte
	if opts.MockOpts.CaptureArgs {
		argsBytes = readFileOrFatal(t, mock.ArgsFile)
	}

	return out, string(argsBytes), "", err
}

func runMockKimiReview(
	t *testing.T, model, prompt string,
	stdoutLines []string,
) (output, args, stdin string) {
	t.Helper()

	if stdoutLines == nil {
		stdoutLines = []string{
			makeKimiEvent("assistant", "ok"),
		}
	}

	out, argsStr, stdinStr, err := executeKimiReviewTest(t, reviewTestOpts{
		MockOpts: MockCLIOpts{
			CaptureArgs: true,
			StdoutLines: stdoutLines,
		},
		Model:  model,
		Prompt: prompt,
	})
	require.NoError(t, err, "Review failed: %v")

	return out, argsStr, stdinStr
}

func makeKimiEvent(role, content string) string {
	ev := map[string]any{"role": role, "content": content}
	b, err := json.Marshal(ev)
	if err != nil {
		panic("makeKimiEvent: " + err.Error())
	}
	return string(b)
}

func makeKimiMetaEvent(metaType, sessionID, command string) string {
	ev := map[string]any{
		"role":       "meta",
		"type":       metaType,
		"session_id": sessionID,
		"command":    command,
		"content":    "To resume this session: " + command,
	}
	b, err := json.Marshal(ev)
	if err != nil {
		panic("makeKimiMetaEvent: " + err.Error())
	}
	return string(b)
}
