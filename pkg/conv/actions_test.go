package conv

import "testing"

func TestGenerateResumeCommand_PiMono(t *testing.T) {
	session := &Session{
		ID:    "018f-session",
		Agent: AgentPiMono,
	}

	command := GenerateResumeCommand(session)
	if command != "pi --session 018f-session" {
		t.Fatalf("unexpected resume command %q", command)
	}
}
