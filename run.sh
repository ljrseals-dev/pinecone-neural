#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  echo "Copy .env.example to .env and fill in your keys first."
  exit 1
fi

export PYTHONPATH=.
exec uvicorn app.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
