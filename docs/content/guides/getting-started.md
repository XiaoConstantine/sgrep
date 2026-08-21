---
title: Getting started
description: Install sgrep, prepare the local embedding model, and run your first code and conversation searches.
weight: 10
---

## Install

Homebrew is the recommended installation method on macOS and Linux:

```bash
brew tap XiaoConstantine/tap
brew install sgrep
```

You can also use the install script or Go:

```bash
# Prebuilt release
curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/sgrep/main/install.sh | bash

# Go toolchain
go install github.com/XiaoConstantine/sgrep/cmd/sgrep@latest
```

To build from source:

```bash
git clone https://github.com/XiaoConstantine/sgrep.git
cd sgrep
go build -o sgrep ./cmd/sgrep
```

Source builds require Go 1.27 or later. Prebuilt Intel macOS binaries support
macOS 13 Ventura or later; Apple Silicon binaries require macOS 15.5 or later.

Go 1.27's experimental portable SIMD kernels can be enabled for source builds
on non-ARM64 systems. ARM64 builds continue to use the faster NEON assembly:

```bash
GOEXPERIMENT=simd go build -o sgrep ./cmd/sgrep
```

The optional sqlite-vec backend is selected at build time:

```bash
go build -tags=sqlite_vec -o sgrep ./cmd/sgrep
```

## Prepare the embedding service

sgrep uses llama.cpp for local embeddings. On macOS, install it with Homebrew:

```bash
brew install llama.cpp
```

Then download the default Nomic embedding model and verify the installation:

```bash
sgrep setup
```

The managed embedding server starts on demand and stays available as a daemon. Use `SGREP_ENDPOINT` instead when you want sgrep to connect to an external compatible server.

## Search a repository

```bash
cd your-repository
sgrep index .
sgrep "database connection error handling"
```

The default `balanced` profile combines semantic and lexical retrieval. Use `fast` for the lowest latency and `quality` for ColBERT late interaction:

```bash
sgrep --profile fast "request routing"
sgrep --profile quality "JWT validation middleware"
```

See [code search](../code-search/) for profiles, watch mode, output formats, and advanced retrieval controls.

## Search coding-agent history

```bash
sgrep conv index
sgrep conv "embedding server ownership bug"
```

sgrep discovers local Claude Code, Codex CLI, Cursor, OpenCode, and Pi histories. See [conversation search](../conversation-search/) for source filters, date filters, export, and resume commands.
