---
title: Agent integration
description: Install the reusable agent skill and combine semantic, structural, and exact search effectively.
weight: 40
---

## Choose the search primitive

sgrep complements rather than replaces structural and exact search:

| Tool | Use it to find | Example |
|------|----------------|---------|
| `sgrep` | Intent and concepts | `sgrep "authentication logic"` |
| `ast-grep` | Syntax structures | `sg -p '$fn($args)'` |
| `ripgrep` | Exact text and regex | `rg "JWT_SECRET"` |

A useful workflow is semantic discovery, then structural matching in the relevant files, then exact symbol lookup.

```bash
sgrep "rate limiting implementation"
sg -p 'rateLimiter.Check($ctx, $key)'
rg "RATE_LIMIT_MAX"
```

## Install the shared skill

Install the canonical repository skill with the cross-agent installer:

```bash
npx skills add XiaoConstantine/sgrep --skill sgrep -g
```

Append `-y` for unattended installation. Use an agent selector such as `--agent claude-code` to target one supported client.

If Node.js is unavailable, sgrep includes an offline fallback:

```bash
sgrep install-skill
```

The fallback writes the skill to `~/.agents/skills/sgrep/SKILL.md` and `~/.claude/skills/sgrep/SKILL.md`. The `npx skills` path is preferred for cross-client installation and updates.

## Claude Code hooks

Claude Code users can also install automatic project indexing, watch hooks, and a Claude-specific copy of the skill:

```bash
sgrep install-claude-code
```

Restart Claude Code after installation. The project hook starts indexing when a session opens and keeps the index updated while you work.

## Integration files

- The repository-root `SKILL.md` is the canonical skill package.
- `plugins/sgrep/skills/sgrep/SKILL.md` is the embedded plugin distribution copy and is checked against the canonical root skill by tests.
- `AGENTS.md` provides repository-level workflow guidance for Codex, Amp, and other compatible agents.
- `.claude/commands/` contains explicit Claude slash commands; commands complement the skill rather than replacing it.
