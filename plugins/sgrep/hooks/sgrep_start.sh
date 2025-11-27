#!/bin/bash
# Start sgrep indexing and watch for the current project

# Check if sgrep is installed
if ! command -v sgrep &> /dev/null; then
    echo "sgrep not found. Install with: brew tap XiaoConstantine/tap && brew install sgrep"
    exit 0
fi

# Get the project root (where Claude Code is running)
PROJECT_ROOT="${CLAUDE_PROJECT_ROOT:-$(pwd)}"

# Check if already indexed
if sgrep status &> /dev/null; then
    echo "sgrep: Index exists, starting watch mode..."
else
    echo "sgrep: Indexing $PROJECT_ROOT..."
    sgrep index "$PROJECT_ROOT" &> /dev/null &
fi

# Start watch mode in background (if not already running)
if ! pgrep -f "sgrep watch" > /dev/null; then
    nohup sgrep watch "$PROJECT_ROOT" &> /dev/null &
    echo "sgrep: Watch mode started"
else
    echo "sgrep: Watch mode already running"
fi
