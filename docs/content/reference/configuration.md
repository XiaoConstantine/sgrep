---
title: Configuration
description: Configure storage, embedding context, managed and external server modes, devices, and vector backends.
weight: 20
---

sgrep works without configuration. Environment variables are intended for alternate storage locations, compatible external embedding servers, and reproducible backend experiments.

## Core settings

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGREP_HOME` | `~/.sgrep` | Models, repository indexes, and managed-server state |
| `SGREP_DIMS` | `768` | Embedding dimensions; changing it requires reindexing |
| `SGREP_CONTEXT_TOKENS` | `512` | Per-slot embedding context; changing it requires reindexing |
| `SGREP_MAX_TOKENS` | context budget minus reserve | Optional source-chunk budget override |

The model format adds `search_query:` to queries and `search_document:` to indexed material. Index metadata records the format and context budget so incompatible indexes fail validation instead of mixing embeddings.

## Managed server mode

With no endpoint override, sgrep owns a local llama.cpp server and starts it when needed.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGREP_PORT` | `8080` | Port for the managed server |
| `SGREP_DEVICE` | automatic | llama.cpp device selection |
| `SGREP_N_GPU_LAYERS` | automatic | llama.cpp GPU layer count |

The manager persists process identity under `SGREP_HOME` and validates the executable, process start identity, port, and embedding capability before treating a process as owned.

```bash
SGREP_PORT=8082 sgrep "authentication"
sgrep server status
sgrep server stop
```

## External endpoint mode

Setting `SGREP_ENDPOINT` opts into an externally managed compatible endpoint. In this mode sgrep validates embedding capability but does not start or stop the process.

```bash
SGREP_ENDPOINT=http://localhost:9090 sgrep index .
```

Do not set both variables expecting `SGREP_PORT` to rewrite the external URL; `SGREP_ENDPOINT` is the complete endpoint override.

## Vector backend overrides

The normal code-search path selects the compatible local artifact automatically. These variables are mainly useful for benchmark isolation and diagnostics:

| Variable | Values | Purpose |
|----------|--------|---------|
| `SGREP_VECTOR_BACKEND` | `tqmse`, `sqlite`, `libsql` | Force compact TQ-MSE or legacy SQL search |
| `SGREP_CONV_VECTOR_BACKEND` | `tqmse`, `sqlite`, `libsql` | Override conversation search independently |

Forcing `sqlite` or `libsql` requires an index built with `sgrep index --sql-vectors`. Forcing `tqmse` fails if the required compact sidecar is absent.

## Reranker settings

| Variable | Purpose |
|----------|---------|
| `SGREP_RERANKER_PORT` | Port for the optional reranker server |
| `SGREP_RERANK_MODEL` | Path to an alternate reranker model |

Install the default reranker model with `sgrep setup --with-rerank` before using `--rerank`.
