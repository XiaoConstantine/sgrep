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

func BenchmarkBuildLexicalText(b *testing.B) {
	content := strings.Repeat(`
func (idx *Indexer) buildColBERTChunkSegments(ctx context.Context, chunks []store.ChunkInfo) error {
	combinedText := util.CombineDescriptionContent(chunk.Content, chunk.Description)
	return segmentStore.StoreColBERTSegmentsBatch(ctx, chunkSegments)
}
`, 8)
	b.ReportAllocs()
	b.SetBytes(int64(len(content)))
	for b.Loop() {
		buildLexicalText("pkg/index/colbert_segments.go", "Go method buildColBERTChunkSegments", content)
	}
}
