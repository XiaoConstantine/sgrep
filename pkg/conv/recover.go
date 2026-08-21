package conv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	RecallSchema          = "sgrep.conv.recover/v1"
	DefaultRecallMaxBytes = 24 * 1024
	MinRecallMaxBytes     = 4 * 1024
	MaxRecallMaxBytes     = 128 * 1024
)

type RecallStatus string

const (
	RecallOK                 RecallStatus = "ok"
	RecallPartial            RecallStatus = "partial"
	RecallNoMatches          RecallStatus = "no_matches"
	RecallNotReady           RecallStatus = "not_ready"
	RecallBackendUnavailable RecallStatus = "backend_unavailable"
	RecallInvalidRequest     RecallStatus = "invalid_request"
	RecallInternalError      RecallStatus = "internal_error"
)

// RecallOptions controls the bounded context returned to a coding agent.
type RecallOptions struct {
	MaxBytes   int
	CurrentDir string
}

type RecallBudget struct {
	MaxBytes  int  `json:"max_bytes"`
	UsedBytes int  `json:"used_bytes"`
	Truncated bool `json:"truncated"`
}

type RecallIndex struct {
	LastIndexed *time.Time `json:"last_indexed,omitempty"`
	Ready       bool       `json:"ready"`
}

type RecallWarning struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type RecallEvidence struct {
	Citation    string   `json:"citation"`
	SourceRef   string   `json:"source_ref"`
	TurnIndex   int      `json:"turn_index"`
	Reasons     []string `json:"reasons"`
	MatchRank   int      `json:"match_rank,omitempty"`
	User        string   `json:"user"`
	Assistant   string   `json:"assistant"`
	ContentHash string   `json:"content_hash"`
	Truncated   bool     `json:"truncated"`
	Untrusted   bool     `json:"untrusted"`
}

type RecallSession struct {
	SourceID    string           `json:"source_id"`
	Agent       AgentType        `json:"agent"`
	SessionID   string           `json:"session_id"`
	ProjectPath string           `json:"project_path,omitempty"`
	ProjectName string           `json:"project_name,omitempty"`
	GitBranch   string           `json:"git_branch,omitempty"`
	GitCommit   string           `json:"git_commit,omitempty"`
	StartedAt   time.Time        `json:"started_at"`
	EndedAt     time.Time        `json:"ended_at,omitempty"`
	Affinity    string           `json:"affinity"`
	Evidence    []RecallEvidence `json:"evidence"`
}

type RecallResponse struct {
	Schema          string          `json:"schema"`
	Status          RecallStatus    `json:"status"`
	Query           string          `json:"query"`
	CurrentProject  string          `json:"current_project,omitempty"`
	RetrievalMode   string          `json:"retrieval_mode"`
	Index           RecallIndex     `json:"index"`
	Budget          RecallBudget    `json:"budget"`
	Sessions        []RecallSession `json:"sessions"`
	OmittedEvidence int             `json:"omitted_evidence"`
	Warnings        []RecallWarning `json:"warnings"`
}

type recallTurnRetriever interface {
	RetrieveTurns(context.Context, string, SearchOptions) ([]SearchResult, error)
}

type recallSessionReader interface {
	GetSession(context.Context, string) (*Session, error)
	GetStats(context.Context) (*IndexStats, error)
}

type recallSnapshot interface {
	ConversationGeneration(context.Context) (int64, error)
	RefreshVectorSnapshot(context.Context) error
}

// RecallService assembles cited, query-aware context from prior conversations.
type RecallService struct {
	turns    recallTurnRetriever
	sessions recallSessionReader
	snapshot recallSnapshot
}

func NewRecallService(store *Store, searcher *Searcher) *RecallService {
	return &RecallService{turns: searcher, sessions: store, snapshot: store}
}

type recallSessionGroup struct {
	session *Session
	matches []rankedRecallTurn
	score   float64
	current bool
}

type rankedRecallTurn struct {
	result SearchResult
	rank   int
}

type evidenceKey struct {
	sessionID string
	turnIndex int
}

type evidencePriority struct {
	priority  int
	session   int
	matchRank int
}

// Recall searches all indexed agents and returns bounded historical evidence.
func (s *RecallService) Recall(ctx context.Context, query string, opts RecallOptions) (*RecallResponse, error) {
	query = strings.TrimSpace(query)
	if opts.MaxBytes == 0 {
		opts.MaxBytes = DefaultRecallMaxBytes
	}
	response := newRecallResponse(query, opts.MaxBytes)
	if query == "" || len(query) > 2048 || opts.MaxBytes < MinRecallMaxBytes || opts.MaxBytes > MaxRecallMaxBytes {
		response.Status = RecallInvalidRequest
		response.Warnings = append(response.Warnings, RecallWarning{Code: "invalid_request", Message: "query must be 1-2048 bytes and max-bytes must be 4096-131072"})
		finalizeRecallBudget(response)
		return response, nil
	}

	cwd, err := canonicalCurrentDir(opts.CurrentDir)
	if err != nil {
		response.Status = RecallInvalidRequest
		response.Warnings = append(response.Warnings, RecallWarning{Code: "invalid_cwd", Message: err.Error()})
		finalizeRecallBudget(response)
		return response, nil
	}
	response.CurrentProject = cwd

	stats, err := s.sessions.GetStats(ctx)
	if err != nil {
		return nil, fmt.Errorf("read conversation index status: %w", err)
	}
	if !stats.LastIndexed.IsZero() {
		lastIndexed := stats.LastIndexed
		response.Index.LastIndexed = &lastIndexed
	}
	response.Index.Ready = stats.TotalSessions > 0
	if stats.TotalSessions == 0 {
		response.Status = RecallNotReady
		response.Warnings = append(response.Warnings, RecallWarning{Code: "index_empty", Message: "conversation index is empty; run sgrep conv index after obtaining user consent"})
		finalizeRecallBudget(response)
		return response, nil
	}

	var generation int64
	if s.snapshot != nil {
		generation, err = s.snapshot.ConversationGeneration(ctx)
		if err != nil {
			return nil, fmt.Errorf("read conversation index generation: %w", err)
		}
	}

	searchOpts := DefaultSearchOptions()
	searchOpts.Limit = 60
	searchOpts.Threshold = 0.35
	searchOpts.UseHybrid = true
	results, searchErr := s.turns.RetrieveTurns(ctx, query, searchOpts)
	if searchErr != nil {
		searchOpts.ExactMatch = true
		searchOpts.UseHybrid = false
		results, err = s.turns.RetrieveTurns(ctx, query, searchOpts)
		if err != nil {
			response.Status = RecallBackendUnavailable
			response.Warnings = append(response.Warnings, RecallWarning{Code: "retrieval_unavailable", Message: "semantic and lexical conversation retrieval failed"})
			finalizeRecallBudget(response)
			return response, nil
		}
		response.Status = RecallPartial
		response.RetrievalMode = "lexical_fallback"
		response.Warnings = append(response.Warnings, RecallWarning{Code: "semantic_fallback", Message: "semantic retrieval failed; results use lexical matching only"})
	}
	groups, inconsistent := s.groupRecallResults(ctx, results, cwd)
	if s.snapshot != nil {
		currentGeneration, generationErr := s.snapshot.ConversationGeneration(ctx)
		if generationErr != nil {
			return nil, fmt.Errorf("verify conversation index generation: %w", generationErr)
		}
		inconsistent = inconsistent || currentGeneration != generation
	}
	if inconsistent {
		// A session was replaced between retrieval and hydration. Refresh any
		// loaded vector sidecar and retry once so stale scores cannot be attached
		// to new turn content.
		if s.snapshot != nil {
			if refreshErr := s.snapshot.RefreshVectorSnapshot(ctx); refreshErr != nil {
				return nil, fmt.Errorf("refresh conversation vector snapshot: %w", refreshErr)
			}
			generation, err = s.snapshot.ConversationGeneration(ctx)
			if err != nil {
				return nil, fmt.Errorf("read refreshed conversation index generation: %w", err)
			}
		}
		results, err = s.turns.RetrieveTurns(ctx, query, searchOpts)
		if err == nil {
			groups, inconsistent = s.groupRecallResults(ctx, results, cwd)
		}
		if s.snapshot != nil && err == nil {
			currentGeneration, generationErr := s.snapshot.ConversationGeneration(ctx)
			if generationErr != nil {
				return nil, fmt.Errorf("verify refreshed conversation index generation: %w", generationErr)
			}
			inconsistent = inconsistent || currentGeneration != generation
		}
		if inconsistent || err != nil {
			// Do not retain first-attempt groups: a generation change means their
			// scores may all have come from the stale sidecar.
			groups = nil
			response.Status = RecallPartial
			response.Warnings = append(response.Warnings, RecallWarning{Code: "index_changed", Message: "the conversation index changed during recall; inconsistent sessions were omitted"})
		}
	}
	if len(groups) == 0 {
		if response.Status != RecallPartial {
			response.Status = RecallNoMatches
		}
		finalizeRecallBudget(response)
		return response, nil
	}
	if len(groups) > 3 {
		groups = groups[:3]
	}

	priorities := make(map[evidenceKey]evidencePriority)
	for i, group := range groups {
		recallSession := buildRecallSession(group, i)
		response.Sessions = append(response.Sessions, recallSession)
		for matchIndex, match := range group.matches {
			priority := 1
			if matchIndex == 0 {
				priority = 0
			}
			addRecallTurn(&response.Sessions[i], group.session, match.result.TurnIndex, "match", match.rank, priority, i, priorities, opts.MaxBytes)
		}
		for _, match := range group.matches {
			position := turnPosition(group.session.Turns, match.result.TurnIndex)
			if position < 0 {
				continue
			}
			if position > 0 {
				addRecallTurn(&response.Sessions[i], group.session, group.session.Turns[position-1].Index, "before", 0, 2, i, priorities, opts.MaxBytes)
			}
			if position+1 < len(group.session.Turns) {
				addRecallTurn(&response.Sessions[i], group.session, group.session.Turns[position+1].Index, "after", 0, 2, i, priorities, opts.MaxBytes)
			}
		}
		completed := completedTurnIndexes(group.session.Turns, 2)
		for _, turnIndex := range completed {
			addRecallTurn(&response.Sessions[i], group.session, turnIndex, "tail", 0, 3, i, priorities, opts.MaxBytes)
		}
	}

	for _, session := range response.Sessions {
		for _, evidence := range session.Evidence {
			if evidence.Truncated {
				response.Budget.Truncated = true
			}
		}
	}
	if response.Budget.Truncated {
		response.Warnings = append(response.Warnings, RecallWarning{Code: "budget_truncated", Message: "historical evidence was omitted or truncated to fit the response budget"})
	}
	if response.RetrievalMode == "lexical_fallback" || response.Budget.Truncated {
		response.Status = RecallPartial
	}
	pruneRecallToBudget(response, priorities)
	if response.OmittedEvidence > 0 || response.Budget.Truncated {
		response.Status = RecallPartial
		// Status is serialized inside the bounded document, so enforce the cap
		// once more after changing it.
		pruneRecallToBudget(response, priorities)
	}
	return response, nil
}

func newRecallResponse(query string, maxBytes int) *RecallResponse {
	return &RecallResponse{
		Schema:        RecallSchema,
		Status:        RecallOK,
		Query:         query,
		RetrievalMode: "hybrid",
		Budget:        RecallBudget{MaxBytes: maxBytes},
		Sessions:      make([]RecallSession, 0),
		Warnings:      make([]RecallWarning, 0),
	}
}

func canonicalCurrentDir(dir string) (string, error) {
	if dir == "" {
		var err error
		dir, err = filepath.Abs(".")
		if err != nil {
			return "", err
		}
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	return filepath.Clean(abs), nil
}

func (s *RecallService) groupRecallResults(ctx context.Context, results []SearchResult, cwd string) ([]recallSessionGroup, bool) {
	bySession := make(map[string]*recallSessionGroup)
	seenTurns := make(map[string]struct{})
	blockedSessions := make(map[string]struct{})
	inconsistent := false
	for i, result := range results {
		if _, blocked := blockedSessions[result.SessionID]; blocked {
			continue
		}
		if _, ok := seenTurns[result.TurnID]; ok {
			continue
		}
		seenTurns[result.TurnID] = struct{}{}
		group := bySession[result.SessionID]
		if group == nil {
			session, err := s.sessions.GetSession(ctx, result.SessionID)
			if err != nil {
				continue
			}
			group = &recallSessionGroup{session: session, current: isCurrentProject(session.ProjectPath, cwd)}
			bySession[result.SessionID] = group
		}
		position := turnPosition(group.session.Turns, result.TurnIndex)
		if position < 0 || group.session.Turns[position].UserContent != result.UserContent || group.session.Turns[position].AssistContent != result.AssistContent {
			inconsistent = true
			blockedSessions[result.SessionID] = struct{}{}
			delete(bySession, result.SessionID)
			continue
		}
		if len(group.matches) >= 3 {
			continue
		}
		rank := i + 1
		group.matches = append(group.matches, rankedRecallTurn{result: result, rank: rank})
		group.score += 1.0 / (60.0 + float64(rank))
	}

	groups := make([]recallSessionGroup, 0, len(bySession))
	for _, group := range bySession {
		if group.current {
			group.score += 0.25 / 61.0
		}
		groups = append(groups, *group)
	}
	sort.SliceStable(groups, func(i, j int) bool {
		if groups[i].score != groups[j].score {
			return groups[i].score > groups[j].score
		}
		if groups[i].session.Agent != groups[j].session.Agent {
			return groups[i].session.Agent < groups[j].session.Agent
		}
		return groups[i].session.ID < groups[j].session.ID
	})
	return groups, inconsistent
}

func isCurrentProject(projectPath, cwd string) bool {
	if projectPath == "" || cwd == "" {
		return false
	}
	project, err := filepath.Abs(projectPath)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(filepath.Clean(project), filepath.Clean(cwd))
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)))
}

func buildRecallSession(group recallSessionGroup, index int) RecallSession {
	affinity := "other"
	if group.current {
		affinity = "current"
	}
	return RecallSession{
		SourceID:    fmt.Sprintf("S%d", index+1),
		Agent:       group.session.Agent,
		SessionID:   group.session.ID,
		ProjectPath: group.session.ProjectPath,
		ProjectName: group.session.ProjectName,
		GitBranch:   group.session.GitBranch,
		GitCommit:   group.session.GitCommit,
		StartedAt:   group.session.StartedAt,
		EndedAt:     group.session.EndedAt,
		Affinity:    affinity,
		Evidence:    make([]RecallEvidence, 0),
	}
}

func addRecallTurn(dst *RecallSession, session *Session, turnIndex int, reason string, matchRank, priority, sessionRank int, priorities map[evidenceKey]evidencePriority, maxBytes int) {
	position := turnPosition(session.Turns, turnIndex)
	if position < 0 {
		return
	}
	key := evidenceKey{sessionID: session.ID, turnIndex: turnIndex}
	for i := range dst.Evidence {
		if dst.Evidence[i].TurnIndex == turnIndex {
			dst.Evidence[i].Reasons = appendReason(dst.Evidence[i].Reasons, reason)
			if reason == "match" && (dst.Evidence[i].MatchRank == 0 || matchRank < dst.Evidence[i].MatchRank) {
				dst.Evidence[i].MatchRank = matchRank
			}
			if old := priorities[key]; priority < old.priority {
				priorities[key] = evidencePriority{priority: priority, session: sessionRank, matchRank: matchRank}
			}
			return
		}
	}

	turn := session.Turns[position]
	perFieldLimit := maxBytes / 5
	if perFieldLimit < 512 {
		perFieldLimit = 512
	}
	if perFieldLimit > 4096 {
		perFieldLimit = 4096
	}
	user, userTruncated := truncateMiddleBytes(turn.UserContent, perFieldLimit)
	assistant, assistantTruncated := truncateMiddleBytes(turn.AssistContent, perFieldLimit)
	hash := sha256.Sum256([]byte(turn.UserContent + "\x00" + turn.AssistContent))
	dst.Evidence = append(dst.Evidence, RecallEvidence{
		SourceRef:   fmt.Sprintf("conv://%s/%s/turn/%d", session.Agent, url.PathEscape(session.ID), turnIndex),
		TurnIndex:   turnIndex,
		Reasons:     []string{reason},
		MatchRank:   matchRank,
		User:        user,
		Assistant:   assistant,
		ContentHash: hex.EncodeToString(hash[:]),
		Truncated:   userTruncated || assistantTruncated,
		Untrusted:   true,
	})
	priorities[key] = evidencePriority{priority: priority, session: sessionRank, matchRank: matchRank}
}

func appendReason(reasons []string, reason string) []string {
	for _, existing := range reasons {
		if existing == reason {
			return reasons
		}
	}
	return append(reasons, reason)
}

func turnPosition(turns []Turn, turnIndex int) int {
	for i := range turns {
		if turns[i].Index == turnIndex {
			return i
		}
	}
	return -1
}

func completedTurnIndexes(turns []Turn, limit int) []int {
	indexes := make([]int, 0, limit)
	for i := len(turns) - 1; i >= 0 && len(indexes) < limit; i-- {
		if strings.TrimSpace(turns[i].AssistContent) != "" {
			indexes = append(indexes, turns[i].Index)
		}
	}
	for i, j := 0, len(indexes)-1; i < j; i, j = i+1, j-1 {
		indexes[i], indexes[j] = indexes[j], indexes[i]
	}
	return indexes
}

func pruneRecallToBudget(response *RecallResponse, priorities map[evidenceKey]evidencePriority) {
	for {
		sortAndCiteRecall(response)
		finalizeRecallBudget(response)
		if response.Budget.UsedBytes <= response.Budget.MaxBytes {
			break
		}
		if !response.Budget.Truncated {
			response.Budget.Truncated = true
			response.Warnings = append(response.Warnings, RecallWarning{Code: "budget_truncated", Message: "historical evidence was omitted or truncated to fit the response budget"})
		}
		sessionIndex, evidenceIndex := worstRecallEvidence(response, priorities)
		if sessionIndex < 0 {
			break
		}
		if totalRecallEvidence(response) == 1 {
			evidence := &response.Sessions[sessionIndex].Evidence[evidenceIndex]
			oldSize := len(evidence.User) + len(evidence.Assistant)
			if oldSize <= 128 {
				response.Sessions[sessionIndex].Evidence = nil
				response.OmittedEvidence++
				response.Budget.Truncated = true
				removeEmptyRecallSessions(response)
				continue
			}
			target := oldSize / 2
			evidence.User, _ = truncateMiddleBytes(evidence.User, target/2)
			evidence.Assistant, _ = truncateMiddleBytes(evidence.Assistant, target/2)
			evidence.Truncated = true
			response.Budget.Truncated = true
			continue
		}
		response.Sessions[sessionIndex].Evidence = append(response.Sessions[sessionIndex].Evidence[:evidenceIndex], response.Sessions[sessionIndex].Evidence[evidenceIndex+1:]...)
		response.OmittedEvidence++
		response.Budget.Truncated = true
		removeEmptyRecallSessions(response)
	}
	finalizeRecallBudget(response)
}

func worstRecallEvidence(response *RecallResponse, priorities map[evidenceKey]evidencePriority) (int, int) {
	bestSession, bestEvidence := -1, -1
	var worst evidencePriority
	for si := range response.Sessions {
		for ei := range response.Sessions[si].Evidence {
			evidence := response.Sessions[si].Evidence[ei]
			p := priorities[evidenceKey{sessionID: response.Sessions[si].SessionID, turnIndex: evidence.TurnIndex}]
			if bestSession == -1 || p.priority > worst.priority ||
				(p.priority == worst.priority && p.session > worst.session) ||
				(p.priority == worst.priority && p.session == worst.session && p.matchRank > worst.matchRank) {
				bestSession, bestEvidence, worst = si, ei, p
			}
		}
	}
	return bestSession, bestEvidence
}

func totalRecallEvidence(response *RecallResponse) int {
	total := 0
	for i := range response.Sessions {
		total += len(response.Sessions[i].Evidence)
	}
	return total
}

func removeEmptyRecallSessions(response *RecallResponse) {
	out := response.Sessions[:0]
	for _, session := range response.Sessions {
		if len(session.Evidence) > 0 {
			out = append(out, session)
		}
	}
	response.Sessions = out
}

func sortAndCiteRecall(response *RecallResponse) {
	citation := 1
	for si := range response.Sessions {
		sort.SliceStable(response.Sessions[si].Evidence, func(i, j int) bool {
			return response.Sessions[si].Evidence[i].TurnIndex < response.Sessions[si].Evidence[j].TurnIndex
		})
		for ei := range response.Sessions[si].Evidence {
			response.Sessions[si].Evidence[ei].Citation = fmt.Sprintf("E%d", citation)
			citation++
		}
	}
}

// EnforceRecallBudget bounds non-evidence fields and records the exact compact
// JSON size, including the newline emitted by json.Encoder. Evidence pruning is
// handled separately, but every early status uses this path as well.
func EnforceRecallBudget(response *RecallResponse) {
	finalizeRecallSize(response)
	if response.Budget.MaxBytes <= 0 || response.Budget.UsedBytes <= response.Budget.MaxBytes {
		return
	}
	response.Budget.Truncated = true
	response.Query, _ = truncateMiddleBytes(response.Query, 512)
	response.CurrentProject, _ = truncateMiddleBytes(response.CurrentProject, 256)
	for i := range response.Warnings {
		response.Warnings[i].Message, _ = truncateMiddleBytes(response.Warnings[i].Message, 256)
	}
	finalizeRecallSize(response)
	if response.Budget.UsedBytes > response.Budget.MaxBytes {
		response.Query = ""
		response.CurrentProject = ""
		finalizeRecallSize(response)
	}
}

func finalizeRecallBudget(response *RecallResponse) {
	EnforceRecallBudget(response)
}

func finalizeRecallSize(response *RecallResponse) {
	for i := 0; i < 8; i++ {
		data, err := json.Marshal(response)
		if err != nil {
			return
		}
		used := len(data) + 1
		if response.Budget.UsedBytes == used {
			return
		}
		response.Budget.UsedBytes = used
	}
}

func truncateMiddleBytes(value string, maxBytes int) (string, bool) {
	if maxBytes <= 0 || len(value) <= maxBytes {
		return value, false
	}
	const marker = "\n…[truncated]…\n"
	if maxBytes <= len(marker)+8 {
		value = value[:maxBytes]
		for !utf8.ValidString(value) && len(value) > 0 {
			value = value[:len(value)-1]
		}
		return value, true
	}
	available := maxBytes - len(marker)
	headEnd := available / 2
	for headEnd > 0 && !utf8.RuneStart(value[headEnd]) {
		headEnd--
	}
	tailStart := len(value) - (available - headEnd)
	for tailStart < len(value) && !utf8.RuneStart(value[tailStart]) {
		tailStart++
	}
	return value[:headEnd] + marker + value[tailStart:], true
}
