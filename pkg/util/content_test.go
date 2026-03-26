package util

import "testing"

func TestCombineDescriptionContent(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		description string
		want        string
	}{
		{
			name:        "combines description and content",
			content:     "func main() {}",
			description: "entrypoint",
			want:        "entrypoint\n\nfunc main() {}",
		},
		{
			name:        "trims whitespace",
			content:     "  func main() {}  ",
			description: "  entrypoint  ",
			want:        "entrypoint\n\nfunc main() {}",
		},
		{
			name:        "description only",
			description: "entrypoint",
			want:        "entrypoint",
		},
		{
			name:    "content only",
			content: "func main() {}",
			want:    "func main() {}",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := CombineDescriptionContent(tc.content, tc.description); got != tc.want {
				t.Fatalf("CombineDescriptionContent() = %q, want %q", got, tc.want)
			}
		})
	}
}
