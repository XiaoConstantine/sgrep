package store

import "testing"

func TestFuseHybridCandidatesUsesRanksAndIncludesLexicalOnly(t *testing.T) {
	dense := []DenseSearchResult{{ID: "dense", Distance: 0.1}, {ID: "both", Distance: 0.2}}
	lexical := []BM25SearchResult{{ID: "lexical", Score: -1000}, {ID: "both", Score: -1}}
	got := fuseHybridCandidates(dense, lexical, 3, 0.5, 0.5)
	if len(got) != 3 {
		t.Fatalf("got %d candidates, want 3", len(got))
	}
	if got[0].ID != "both" {
		t.Fatalf("top candidate = %q, want candidate present in both rankings", got[0].ID)
	}
	seenLexical := false
	for _, hit := range got {
		if hit.ID == "lexical" {
			seenLexical = true
		}
	}
	if !seenLexical {
		t.Fatal("lexical-only candidate was dropped")
	}
}

func TestFuseHybridCandidatesHonorsWeights(t *testing.T) {
	dense := []DenseSearchResult{{ID: "dense", Distance: 0.9}}
	lexical := []BM25SearchResult{{ID: "lexical", Score: -0.01}}
	got := fuseHybridCandidates(dense, lexical, 2, 0.9, 0.1)
	if len(got) != 2 || got[0].ID != "dense" {
		t.Fatalf("weighted fusion = %+v, want dense first", got)
	}
}
