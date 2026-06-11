package conv

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	searchterms "github.com/XiaoConstantine/sgrep/pkg/search"
)

// Searcher handles conversation search with embedding integration.
type Searcher struct {
	store    *Store
	embedder *embed.Embedder
}

// NewSearcher creates a new conversation searcher.
func NewSearcher(store *Store, embedder *embed.Embedder) *Searcher {
	return &Searcher{
		store:    store,
		embedder: embedder,
	}
}

// Search performs semantic search on conversations.
func (s *Searcher) Search(ctx context.Context, query string, opts SearchOptions) (*SearchResponse, error) {
	startTime := time.Now()

	// Perform search based on options
	var results []SearchResult
	var err error
	if opts.ExactMatch {
		searchOpts := opts
		searchOpts.Limit = opts.Limit * 2
		results, err = s.store.KeywordSearch(ctx, searchterms.ExtractHybridSearchTerms(query), searchOpts)
	} else {
		if s.embedder == nil {
			return nil, fmt.Errorf("semantic search requires an embedder")
		}

		// Generate query embedding
		queryEmb, err := s.embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("failed to generate query embedding: %w", err)
		}

		hasFilters := opts.Agent != AgentAll || opts.Project != "" || !opts.Since.IsZero() || !opts.Before.IsZero()
		if opts.UseHybrid && hasFilters {
			results, err = s.store.FilteredHybridSearch(ctx, queryEmb, searchterms.ExtractHybridSearchTerms(query), opts)
		} else if opts.UseHybrid {
			results, err = s.store.HybridSearch(ctx, queryEmb, searchterms.ExtractHybridSearchTerms(query), opts.Limit*2, opts.Threshold, opts.SemanticWeight, opts.BM25Weight)
		} else if hasFilters {
			results, err = s.store.FilteredSearch(ctx, queryEmb, opts)
		} else {
			results, err = s.store.VectorSearch(ctx, queryEmb, opts.Limit*2, opts.Threshold)
		}
	}

	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Deduplicate by session (show best turn per session)
	dedupResults := deduplicateBySession(results)

	// Convert to ConversationHit
	hits, err := s.resultsToHits(ctx, dedupResults, opts.Limit)
	if err != nil {
		return nil, err
	}

	// Get index metadata
	stats, _ := s.store.GetStats(ctx)
	metadata := IndexMetadata{
		IndexVersion:    "1.0.0",
		IndexedSessions: stats.TotalSessions,
		LastIndexed:     stats.LastIndexed,
	}
	for agent := range stats.SessionsByAgent {
		metadata.AgentsIndexed = append(metadata.AgentsIndexed, agent)
	}

	return &SearchResponse{
		Query:      query,
		Filters:    opts,
		SearchTime: time.Since(startTime).Milliseconds(),
		TotalHits:  len(results),
		Returned:   len(hits),
		Results:    hits,
		Metadata:   metadata,
	}, nil
}

// deduplicateBySession keeps only the best match per session.
// Scores are similarity values where higher is better.
func deduplicateBySession(results []SearchResult) []SearchResult {
	seen := make(map[string]int) // session ID -> index in deduplicated slice
	var deduped []SearchResult

	for _, r := range results {
		if idx, exists := seen[r.SessionID]; exists {
			// Keep the one with higher score (better match)
			if r.Score > deduped[idx].Score {
				deduped[idx] = r
			}
		} else {
			seen[r.SessionID] = len(deduped)
			deduped = append(deduped, r)
		}
	}

	return deduped
}

// resultsToHits converts raw search results to ConversationHit objects.
func (s *Searcher) resultsToHits(ctx context.Context, results []SearchResult, limit int) ([]ConversationHit, error) {
	var hits []ConversationHit

	for _, r := range results {
		if len(hits) >= limit {
			break
		}

		// Get full session for metadata
		session, err := s.store.GetSession(ctx, r.SessionID)
		if err != nil {
			continue // Skip on error
		}

		// Create preview
		userSnip := truncate(r.UserContent, 80)
		assistSnip := truncate(r.AssistContent, 100)

		hit := ConversationHit{
			SessionID:    r.SessionID,
			Agent:        session.Agent,
			ProjectPath:  session.ProjectPath,
			ProjectName:  session.ProjectName,
			StartedAt:    session.StartedAt,
			RelativeTime: relativeTime(session.StartedAt),
			MatchedTurn: TurnPreview{
				TurnIndex:  r.TurnIndex,
				UserSnip:   userSnip,
				AssistSnip: assistSnip,
				FullUser:   r.UserContent,
				FullAssist: r.AssistContent,
			},
			TotalTurns:  len(session.Turns),
			TotalTokens: session.TotalTokens,
			Score:       r.Score,
			MatchType:   matchTypeOrDefault(r.MatchType),
			Actions:     generateActions(session),
		}

		hits = append(hits, hit)
	}

	return hits, nil
}

func matchTypeOrDefault(matchType string) string {
	if matchType == "" {
		return "semantic"
	}
	return matchType
}

// generateActions creates actionable commands for a session.
func generateActions(session *Session) HitActions {
	return HitActions{
		View:    fmt.Sprintf("sgrep conv view %s", session.ID),
		Export:  fmt.Sprintf("sgrep conv export %s -o conversation.md", session.ID),
		Context: fmt.Sprintf("sgrep conv context %s", session.ID),
		Resume:  GenerateResumeCommand(session),
	}
}

// truncate truncates a string to maxLen characters, adding ellipsis if needed.
func truncate(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	// Remove newlines for preview
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ") // Collapse whitespace

	if len(s) <= maxLen {
		return s
	}

	// Try to break at word boundary
	truncated := s[:maxLen]
	if lastSpace := strings.LastIndex(truncated, " "); lastSpace > maxLen/2 {
		truncated = truncated[:lastSpace]
	}

	return truncated + "..."
}

// relativeTime returns a human-readable relative time string.
func relativeTime(t time.Time) string {
	duration := time.Since(t)

	switch {
	case duration < time.Minute:
		return "just now"
	case duration < time.Hour:
		mins := int(duration.Minutes())
		if mins == 1 {
			return "1 minute ago"
		}
		return fmt.Sprintf("%d minutes ago", mins)
	case duration < 24*time.Hour:
		hours := int(duration.Hours())
		if hours == 1 {
			return "1 hour ago"
		}
		return fmt.Sprintf("%d hours ago", hours)
	case duration < 7*24*time.Hour:
		days := int(duration.Hours() / 24)
		if days == 1 {
			return "1 day ago"
		}
		return fmt.Sprintf("%d days ago", days)
	case duration < 30*24*time.Hour:
		weeks := int(duration.Hours() / 24 / 7)
		if weeks == 1 {
			return "1 week ago"
		}
		return fmt.Sprintf("%d weeks ago", weeks)
	case duration < 365*24*time.Hour:
		months := int(duration.Hours() / 24 / 30)
		if months == 1 {
			return "1 month ago"
		}
		return fmt.Sprintf("%d months ago", months)
	default:
		years := int(duration.Hours() / 24 / 365)
		if years == 1 {
			return "1 year ago"
		}
		return fmt.Sprintf("%d years ago", years)
	}
}

// ParseDuration parses a duration string like "1h", "7d", "2w", "1m", "1y".
func ParseDuration(s string) (time.Duration, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return 0, nil
	}

	// Handle standard Go durations
	if d, err := time.ParseDuration(s); err == nil {
		return d, nil
	}

	// Handle custom formats
	if len(s) < 2 {
		return 0, fmt.Errorf("invalid duration: %s", s)
	}

	unit := s[len(s)-1]
	valueStr := s[:len(s)-1]
	var value int
	if _, err := fmt.Sscanf(valueStr, "%d", &value); err != nil {
		return 0, fmt.Errorf("invalid duration value: %s", s)
	}

	switch unit {
	case 'd':
		return time.Duration(value) * 24 * time.Hour, nil
	case 'w':
		return time.Duration(value) * 7 * 24 * time.Hour, nil
	case 'm':
		return time.Duration(value) * 30 * 24 * time.Hour, nil // Approximate month
	case 'y':
		return time.Duration(value) * 365 * 24 * time.Hour, nil // Approximate year
	default:
		return 0, fmt.Errorf("unknown duration unit: %c", unit)
	}
}
