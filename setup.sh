#!/usr/bin/env bash
set -e

echo "==> ClaimCheck — Environment Setup"

# 1. Python venv
if [ ! -d "backend/.venv" ]; then
  echo "--> Creating Python virtual environment…"
  python3 -m venv backend/.venv
fi

echo "--> Installing Python dependencies…"
source backend/.venv/bin/activate
pip install --upgrade pip -q
pip install -r backend/requirements.txt -q

# 2. .env check
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "--> Created .env from .env.example — add your OPENAI_API_KEY before starting."
fi

# 3. ChromaDB data dir
mkdir -p data/chroma_db

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit .env and set OPENAI_API_KEY=sk-..."
echo "  2. cd backend && source .venv/bin/activate"
echo "  3. uvicorn app.main:app --reload  (backend on :8000)"
echo "  4. Open frontend/index.html in your browser"
