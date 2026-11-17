#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  echo "Copy .env.example to .env and set provider credentials and a unique API_KEYS value."
  exit 1
fi

export PYTHONPATH=.
exec uvicorn app.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
