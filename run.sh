# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  echo "Copy .env.example to .env and set provider credentials and a unique API_KEYS value."
  exit 1
fi

export PYTHONPATH=.
exec python -c 'from app.config import settings; import uvicorn; uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, log_level=settings.log_level.lower())'
