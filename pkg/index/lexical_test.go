package index

import (
	"strings"
	"testing"
)

func TestBuildLexicalTextSplitsCodeIdentifiersAndPaths(t *testing.T) {
	got := buildLexicalText("internal/http_server/JWTValidator.go", "Go function ValidateJWTToken", "func parseOAuth2Response() {}")
	for _, term := range []string{"jwtvalidator", "jwt", "validator", "validate", "token", "http", "server", "oauth2", "response"} {
		if !strings.Contains(" "+got+" ", " "+term+" ") {
			t.Errorf("lexical text %q missing %q", got, term)
		}
	}
}
