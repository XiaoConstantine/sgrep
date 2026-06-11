//go:build sqlite_vec
// +build sqlite_vec

package store

import (
	"context"
	"database/sql"
)

func staleDocumentIDs(ctx context.Context, tx *sql.Tx) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id FROM documents WHERE id NOT IN (SELECT id FROM _sgrep_live_doc_ids)`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}
