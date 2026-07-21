package modelcfg

import (
	"strings"
	"testing"
)

func TestContextBudgets(t *testing.T) {
	for _, value := range []string{"256", "512"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("SGREP_CONTEXT_TOKENS", value)
			if got := ContextTokens(); got != mustAtoi(value) {
				t.Fatalf("ContextTokens = %d, want %s", got, value)
			}
			if DocumentTokenBudget() >= ContextTokens() {
				t.Fatalf("document budget %d must reserve prefix space below context %d", DocumentTokenBudget(), ContextTokens())
			}
		})
	}
}

func TestValidateInputRejectsInsteadOfTruncating(t *testing.T) {
	t.Setenv("SGREP_CONTEXT_TOKENS", "128")
	if err := ValidateInput("short query"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateInput(strings.Repeat("code ", 500)); err == nil {
		t.Fatal("oversized input unexpectedly accepted")
	}
}

func mustAtoi(value string) int {
	var result int
	for _, r := range value {
		result = result*10 + int(r-'0')
	}
	return result
}
