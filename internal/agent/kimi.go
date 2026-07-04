package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// KimiAgent runs code reviews using the Kimi Code CLI.
type KimiAgent struct {
	Command   string         // The kimi command to run (default: "kimi")
	Model     string         // Model alias to use
	Reasoning ReasoningLevel // Reasoning level (preserved for API consistency)
	Agentic   bool           // Whether agentic mode is enabled (auto-approve actions)
	SessionID string         // Existing session ID to resume
}

// NewKimiAgent creates a new Kimi Code agent.
func NewKimiAgent(command string) *KimiAgent {
	if command == "" {
		command = "kimi"
	}
	return &KimiAgent{Command: command, Reasoning: ReasoningStandard}
}

func (a *KimiAgent) clone(opts ...agentCloneOption) *KimiAgent {
	cfg := newAgentCloneConfig(
		a.Command,
		a.Model,
		a.Reasoning,
		a.Agentic,
		a.SessionID,
		opts...,
	)
	return &KimiAgent{
		Command:   cfg.Command,
		Model:     cfg.Model,
		Reasoning: cfg.Reasoning,
		Agentic:   cfg.Agentic,
		SessionID: cfg.SessionID,
	}
}

// WithReasoning returns a copy of the agent with the specified reasoning level.
// Kimi does not expose a reasoning-effort CLI flag, so the level is preserved
// but does not change the emitted command line.
func (a *KimiAgent) WithReasoning(level ReasoningLevel) Agent {
	return a.clone(withClonedReasoning(level))
}

// WithAgentic returns a copy of the agent configured for agentic mode.
func (a *KimiAgent) WithAgentic(agentic bool) Agent {
	return a.clone(withClonedAgentic(agentic))
}

// WithModel returns a copy of the agent configured to use the specified model.
func (a *KimiAgent) WithModel(model string) Agent {
	if model == "" {
		return a
	}
	return a.clone(withClonedModel(model))
}

// WithSessionID returns a copy of the agent configured to resume a prior session.
func (a *KimiAgent) WithSessionID(sessionID string) Agent {
	return a.clone(withClonedSessionID(sessionID))
}

func (a *KimiAgent) Name() string {
	return "kimi"
}

func (a *KimiAgent) CommandName() string {
	return a.Command
}

// buildArgs returns the argv for a Kimi review invocation.
// The prompt is passed via a file reference (@path) because Kimi's -p flag
// requires the prompt as an argument and does not read it from stdin.
//
// Kimi's CLI rejects combining --prompt (-p) with --yolo or --auto, so
// non-interactive agentic mode is not supported. The Agentic field is
// preserved for API consistency but does not change the emitted command line.
func (a *KimiAgent) buildArgs(promptFile string) []string {
	sessionID := sanitizedResumeSessionID(a.SessionID)
	args := []string{"-p", "@" + promptFile, "--output-format", "stream-json"}
	if sessionID != "" {
		args = append(args, "-S", sessionID)
	}
	if a.Model != "" {
		args = append(args, "--model", a.Model)
	}
	return args
}

func (a *KimiAgent) CommandLine() string {
	// CommandLine is used for logging/debugging; it cannot reference a real
	// prompt file, so use a representative placeholder.
	args := a.buildArgs("<prompt-file>")
	return a.Command + " " + strings.Join(args, " ")
}

func (a *KimiAgent) Review(
	ctx context.Context,
	repoPath, commitSHA, prompt string,
	output io.Writer,
) (string, error) {
	// Kimi's -p flag accepts the prompt as an argument, not stdin. To avoid
	// command-line length limits (especially on Windows), write the prompt to a
	// temporary file and pass it by reference.
	promptFile, err := os.CreateTemp("", "roborev-kimi-prompt-*.txt")
	if err != nil {
		return "", fmt.Errorf("create kimi prompt file: %w", err)
	}
	promptPath := promptFile.Name()
	if _, err := promptFile.WriteString(prompt); err != nil {
		_ = promptFile.Close()
		_ = os.Remove(promptPath)
		return "", fmt.Errorf("write kimi prompt file: %w", err)
	}
	if err := promptFile.Close(); err != nil {
		_ = os.Remove(promptPath)
		return "", fmt.Errorf("close kimi prompt file: %w", err)
	}
	defer os.Remove(promptPath)

	args := a.buildArgs(promptPath)

	runResult, runErr := runStreamingCLI(ctx, streamingCLISpec{
		Name:    "kimi",
		Command: a.Command,
		Args:    args,
		Dir:     repoPath,
		Output:  output,
		Parse:   parseKimiJSON,
	})
	if runErr != nil {
		return "", runErr
	}

	if runResult.WaitErr != nil {
		return "", formatDetailedCLIWaitError(runResult, detailedCLIWaitErrorOptions{
			AgentName:     a.Name(),
			Stderr:        runResult.Stderr,
			PartialOutput: runResult.Result,
		})
	}

	if runResult.ParseErr != nil {
		if errors.Is(runResult.ParseErr, errNoKimiJSON) {
			return "", fmt.Errorf("kimi CLI did not emit valid stream-json events; upgrade kimi or check CLI compatibility: %w", errNoKimiJSON)
		}
		return runResult.Result, runResult.ParseErr
	}

	if runResult.Result == "" {
		return "No review output generated", nil
	}
	return runResult.Result, nil
}

// kimiEvent represents a JSONL event from kimi --output-format stream-json.
type kimiEvent struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// errNoKimiJSON indicates no valid kimi stream-json events were parsed.
var errNoKimiJSON = errors.New("no valid kimi stream-json events parsed from output")

// parseKimiJSON parses Kimi's stream-json output and extracts the final
// assistant message content. Meta events (session resume hints) are ignored.
// Tool events reset the accumulated result so only assistant text emitted
// after the last tool call is retained, matching the behavior of other
// agents that drop pre-tool narration.
func parseKimiJSON(r io.Reader, sw *syncWriter) (string, error) {
	var result string
	var validEventsParsed bool

	err := scanStreamJSONLines(r, sw, func(line string) error {
		var ev kimiEvent
		if jsonErr := json.Unmarshal([]byte(line), &ev); jsonErr == nil && ev.Role != "" {
			validEventsParsed = true
			switch ev.Role {
			case "assistant":
				if ev.Content != "" {
					result = stripTerminalControls(ev.Content)
				}
			case "tool":
				// Drop any assistant text that appeared before this tool call.
				result = ""
			}
		}
		return nil
	})
	if err != nil {
		return result, err
	}

	if !validEventsParsed {
		return "", errNoKimiJSON
	}

	return result, nil
}

func init() {
	Register(NewKimiAgent(""))
}
