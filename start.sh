#!/bin/bash
# CascadeIQ — One-command startup
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_BIN="/tmp/node-v20.18.0-darwin-arm64/bin"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║        CascadeIQ — Starting Up           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Kill any existing servers
pkill -f "uvicorn backend.main" 2>/dev/null || true
pkill -f "vite"                 2>/dev/null || true
sleep 1

# Start FastAPI backend
echo "▶ Starting FastAPI backend on http://localhost:8000 ..."
cd "$ROOT"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "  Waiting for models to load..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✓ Backend ready"
    break
  fi
  sleep 1
done

# Start React frontend
echo ""
echo "▶ Starting React frontend on http://localhost:5173 ..."
export PATH="$NODE_BIN:$PATH"
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  CascadeIQ is running!                   ║"
echo "║                                          ║"
echo "║  Frontend : http://localhost:5173        ║"
echo "║  Backend  : http://localhost:8000        ║"
echo "║  API docs : http://localhost:8000/docs   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop all servers."

wait $BACKEND_PID $FRONTEND_PID
