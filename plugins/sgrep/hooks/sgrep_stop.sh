#!/bin/bash
# Stop sgrep watch mode and embedding server

# Stop watch process
if pgrep -f "sgrep watch" > /dev/null; then
    pkill -f "sgrep watch"
    echo "sgrep: Watch mode stopped"
fi

# Optionally stop the embedding server to free resources
# Uncomment if you want to stop the server on session end:
# sgrep server stop &> /dev/null
