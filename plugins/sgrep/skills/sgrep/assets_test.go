package sgrepskill

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

type skillFrontmatter struct {
	Name          string            `yaml:"name"`
	Description   string            `yaml:"description"`
	License       string            `yaml:"license"`
	Compatibility string            `yaml:"compatibility"`
	Metadata      map[string]string `yaml:"metadata"`
}

func TestContentHasValidSkillFrontmatter(t *testing.T) {
	if len(strings.Split(Content, "\n")) > 500 {
		t.Fatalf("skill is too long: %d lines", len(strings.Split(Content, "\n")))
	}

	parts := strings.SplitN(Content, "---\n", 3)
	if len(parts) < 3 || parts[0] != "" {
		t.Fatal("skill content must start with YAML frontmatter")
	}

	var fm skillFrontmatter
	if err := yaml.Unmarshal([]byte(parts[1]), &fm); err != nil {
		t.Fatalf("unmarshal frontmatter: %v", err)
	}

	if fm.Name != "sgrep" {
		t.Fatalf("name = %q, want sgrep", fm.Name)
	}
	if fm.Description == "" {
		t.Fatal("description must not be empty")
	}
	if !strings.Contains(strings.ToLower(fm.Description), "search") || !strings.Contains(strings.ToLower(fm.Description), "use when") {
		t.Fatalf("description should say what it does and when to use it: %q", fm.Description)
	}
	if fm.License != "Apache-2.0" {
		t.Fatalf("license = %q, want Apache-2.0", fm.License)
	}
	if fm.Compatibility == "" {
		t.Fatal("compatibility must not be empty")
	}
	if fm.Metadata["homepage"] == "" {
		t.Fatal("metadata.homepage must not be empty")
	}
	if strings.TrimSpace(parts[2]) == "" {
		t.Fatal("skill body must not be empty")
	}
}

func TestPluginSkillMatchesCanonical(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	skillDir := filepath.Dir(file)
	repoRoot := filepath.Clean(filepath.Join(skillDir, "..", "..", "..", ".."))

	rootSkill, err := os.ReadFile(filepath.Join(repoRoot, "SKILL.md"))
	if err != nil {
		t.Fatalf("read root skill: %v", err)
	}
	if string(rootSkill) != Content {
		t.Fatal("embedded plugin skill drifted from canonical root skill")
	}
}
