// Package parser provides parsers for different coding agent conversation formats.
package parser

import (
	"bufio"
	"bytes"
	"io"
	"os"

	"github.com/XiaoConstantine/sgrep/pkg/conv"
)

// Parser is the interface that all conversation parsers must implement.
type Parser interface {
	// Parse reads conversations from a source path and returns sessions.
	// The source path can be a file or directory depending on the agent format.
	Parse(sourcePath string) ([]*conv.Session, error)

	// Discover finds all conversation files for this agent type.
	// Returns a list of paths that can be passed to Parse.
	Discover() ([]string, error)

	// AgentType returns the agent type this parser handles.
	AgentType() conv.AgentType

	// DefaultPath returns the default search path for this agent's conversations.
	DefaultPath() string
}

// Registry holds all registered parsers.
var Registry = make(map[conv.AgentType]Parser)

// Register adds a parser to the registry.
func Register(p Parser) {
	Registry[p.AgentType()] = p
}

// Get returns a parser for the given agent type.
func Get(agent conv.AgentType) (Parser, bool) {
	p, ok := Registry[agent]
	return p, ok
}

// All returns all registered parsers.
func All() []Parser {
	parsers := make([]Parser, 0, len(Registry))
	for _, p := range Registry {
		parsers = append(parsers, p)
	}
	return parsers
}

// DiscoverAll discovers conversations from all registered parsers.
func DiscoverAll() (map[conv.AgentType][]string, error) {
	result := make(map[conv.AgentType][]string)
	for agent, parser := range Registry {
		paths, err := parser.Discover()
		if err != nil {
			// Continue with other parsers on error
			continue
		}
		if len(paths) > 0 {
			result[agent] = paths
		}
	}
	return result, nil
}

// ParseAll parses conversations from all discovered sources.
func ParseAll() ([]*conv.Session, error) {
	var allSessions []*conv.Session

	for _, parser := range Registry {
		paths, err := parser.Discover()
		if err != nil {
			continue
		}

		for _, path := range paths {
			sessions, err := parser.Parse(path)
			if err != nil {
				continue
			}
			allSessions = append(allSessions, sessions...)
		}
	}

	return allSessions, nil
}

// forEachJSONLLine streams a JSONL file line by line without the token-size
// limit imposed by bufio.Scanner.
func forEachJSONLLine(file *os.File, fn func([]byte) error) error {
	reader := bufio.NewReader(file)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			line = bytes.TrimRight(line, "\r\n")
			if len(line) > 0 {
				if err := fn(line); err != nil {
					return err
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}
