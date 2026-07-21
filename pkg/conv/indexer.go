package conv

import (
	"context"
	"fmt"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
)

// Indexer handles indexing conversations.
type Indexer struct {
	store    *Store
	embedder *embed.Embedder
	chunker  *Chunker
	force    bool
}

// IndexerConfig configures the indexer.
type IndexerConfig struct {
	Store    *Store
	Embedder *embed.Embedder
	Chunker  *Chunker
	Force    bool // Re-index even if session exists
}

// NewIndexer creates a new conversation indexer.
func NewIndexer(cfg IndexerConfig) *Indexer {
	chunker := cfg.Chunker
	if chunker == nil {
		chunker = NewChunker()
	}

	return &Indexer{
		store:    cfg.Store,
		embedder: cfg.Embedder,
		chunker:  chunker,
		force:    cfg.Force,
	}
}

// IndexResult contains the results of an indexing operation.
type IndexResult struct {
	Agent           AgentType
	SessionsFound   int
	SessionsIndexed int
	TurnsIndexed    int
	Errors          []error
	Duration        time.Duration
}

// IndexSessions indexes a batch of parsed sessions.
func (idx *Indexer) IndexSessions(ctx context.Context, sessions []*Session) (*IndexResult, error) {
	startTime := time.Now()
	result := &IndexResult{}

	if len(sessions) > 0 {
		result.Agent = sessions[0].Agent
	}

	for _, session := range sessions {
		result.SessionsFound++

		// Check if already indexed (skip if not forcing)
		if !idx.force {
			exists, _ := idx.store.SessionExists(ctx, session.ID)
			if exists {
				meta, ok, err := idx.store.GetSessionMeta(ctx, session.ID)
				if err != nil {
					result.Errors = append(result.Errors, fmt.Errorf("check session meta for %s failed: %w", session.ID, err))
					continue
				}
				needsUpdate := false
				if ok {
					if len(session.Turns) > meta.TotalTurns {
						needsUpdate = true
					}
					if !session.EndedAt.IsZero() && (meta.EndedAt.IsZero() || session.EndedAt.After(meta.EndedAt)) {
						needsUpdate = true
					}
				}
				if needsUpdate {
					// Fall through to re-index updated session.
				} else {
					missing, err := idx.store.MissingEmbeddingsCountForSession(ctx, session.ID)
					if err != nil {
						result.Errors = append(result.Errors, fmt.Errorf("check missing embeddings for %s failed: %w", session.ID, err))
						continue
					}
					if missing == 0 {
						continue
					}
				}
			}
		}

		// Index the session
		if err := idx.indexSession(ctx, session); err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("index session %s failed: %w", session.ID, err))
			continue
		}

		result.SessionsIndexed++
		result.TurnsIndexed += len(session.Turns)
	}

	result.Duration = time.Since(startTime)
	return result, nil
}

// IndexSession indexes a single session.
func (idx *Indexer) IndexSession(ctx context.Context, session *Session) error {
	// Check if already indexed
	exists, _ := idx.store.SessionExists(ctx, session.ID)
	if exists {
		return nil
	}

	return idx.indexSession(ctx, session)
}

// RebuildTQVectorStore refreshes the compact TQ-MSE sidecar for conversation search.
func (idx *Indexer) RebuildTQVectorStore(ctx context.Context) (int, error) {
	return idx.store.RebuildTQVectorStore(ctx)
}

// indexSession indexes a single session and its turns.
func (idx *Indexer) indexSession(ctx context.Context, session *Session) error {
	// Estimate tokens
	session.TotalTokens = EstimateSessionTokens(session)

	// Store session metadata and turns
	if err := idx.store.StoreSession(ctx, session); err != nil {
		return fmt.Errorf("failed to store session: %w", err)
	}

	// Generate and store embeddings for each turn
	chunks := idx.chunker.ChunkSession(session)

	// Skip embedding if no embedder
	if idx.embedder == nil {
		return nil
	}

	// Batch chunk embeddings, then mean-pool split chunks back to the canonical
	// turn ID used by conv_turns. Storing chunk-suffixed IDs would make long-turn
	// embeddings unreachable by the search joins.
	type turnEmbedding struct {
		sum   []float32
		count int
	}
	pooled := make(map[string]*turnEmbedding)
	batchSize := 10
	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]

		contents := make([]string, len(batch))
		for j, chunk := range batch {
			contents[j] = chunk.Content
		}

		// Generate embeddings
		embeddings, err := idx.embedder.EmbedDocuments(ctx, contents)
		if err != nil {
			return fmt.Errorf("failed to generate embeddings: %w", err)
		}

		for j, embedding := range embeddings {
			turnID := fmt.Sprintf("%s:%d", batch[j].SessionID, batch[j].TurnIndex)
			entry := pooled[turnID]
			if entry == nil {
				entry = &turnEmbedding{sum: make([]float32, len(embedding))}
				pooled[turnID] = entry
			}
			for dim, value := range embedding {
				entry.sum[dim] += value
			}
			entry.count++
		}
	}

	turnIDs := make([]string, 0, len(pooled))
	embeddings := make([][]float32, 0, len(pooled))
	for turnID, entry := range pooled {
		if entry.count == 0 {
			continue
		}
		for dim := range entry.sum {
			entry.sum[dim] /= float32(entry.count)
		}
		turnIDs = append(turnIDs, turnID)
		embeddings = append(embeddings, entry.sum)
	}
	if err := idx.store.StoreTurnEmbeddingBatch(ctx, turnIDs, embeddings); err != nil {
		return fmt.Errorf("failed to store pooled turn embeddings: %w", err)
	}
	return nil
}
