package agent

import (
	"context"
	"encoding/json"
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
func (a *KimiAgent) buildArgs(promptFile string) []string {
	sessionID := sanitizedResumeSessionID(a.SessionID)
	args := []string{"-p", "@" + promptFile, "--output-format", "stream-json"}
	if sessionID != "" {
		args = append(args, "-S", sessionID)
	}
	if a.Model != "" {
		args = append(args, "--model", a.Model)
	}
	if a.Agentic || AllowUnsafeAgents() {
		args = append(args, "--yolo")
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

// parseKimiJSON parses Kimi's stream-json output and extracts the final
// assistant message content. Meta events (session resume hints) and tool
// events are ignored.
func parseKimiJSON(r io.Reader, sw *syncWriter) (string, error) {
	var result string

	err := scanStreamJSONLines(r, sw, func(line string) error {
		var ev kimiEvent
		if jsonErr := json.Unmarshal([]byte(line), &ev); jsonErr == nil {
			if ev.Role == "assistant" && ev.Content != "" {
				result = stripTerminalControls(ev.Content)
			}
		}
		return nil
	})
	if err != nil {
		return result, err
	}

	return result, nil
}

func init() {
	Register(NewKimiAgent(""))
}
