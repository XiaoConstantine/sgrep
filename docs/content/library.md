---
title: Library API
description: Embed semantic code indexing and search in a Go program.
weight: 50
---

Install the Go module:

```bash
go get github.com/XiaoConstantine/sgrep@latest
```

The root package exposes a small semantic code-search client:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/XiaoConstantine/sgrep"
)

func main() {
	ctx := context.Background()

	client, err := sgrep.New("/path/to/codebase")
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	if err := client.Index(ctx); err != nil {
		log.Fatal(err)
	}

	results, err := client.Search(ctx, "authentication logic", 10)
	if err != nil {
		log.Fatal(err)
	}

	for _, result := range results {
		fmt.Printf("%s:%d-%d (score: %.2f)\n",
			result.FilePath, result.StartLine, result.EndLine, result.Score)
	}
}
```

`SearchWithThreshold` exposes the cosine-distance cutoff, and `Watch` blocks while incrementally indexing until its context is cancelled.

For deeper control, use the packages under `pkg/`:

| Package | Responsibility |
|---------|----------------|
| `pkg/index` | Repository indexing and file watching |
| `pkg/search` | Retrieval, fusion, caching, ColBERT, and reranking |
| `pkg/embed` | Local embedding requests and server selection |
| `pkg/store` | SQL metadata and vector artifacts |
| `pkg/chunk` | AST-aware and token-aware source chunking |
| `pkg/conv` | Conversation parsing, indexing, retrieval, and actions |

The CLI is the stable user-facing integration surface. Consumers of `pkg/` should expect a lower-level API that can evolve with the storage and retrieval implementation.
