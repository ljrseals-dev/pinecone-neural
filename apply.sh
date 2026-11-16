#!/usr/bin/env bash
# Apply v2.3 hardening to a pinecone-neural checkout.
# Usage: ./apply.sh /path/to/pinecone-neural

set -euo pipefail

TARGET="${1:-.}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$TARGET/app/neural" ]]; then
  echo "ERROR: $TARGET/app/neural not found. Pass the repo root."
  exit 1
fi

echo "Applying v2.3 patch to $TARGET"

cp "$PATCH_DIR/pipeline.py"    "$TARGET/app/neural/pipeline.py"
cp "$PATCH_DIR/sidecar.py"     "$TARGET/app/neural/sidecar.py"
cp "$PATCH_DIR/synaptic.py"    "$TARGET/app/neural/synaptic.py"
cp "$PATCH_DIR/utility.py"     "$TARGET/app/neural/utility.py"
cp "$PATCH_DIR/neural_init.py" "$TARGET/app/neural/__init__.py"
cp "$PATCH_DIR/harness.py"     "$TARGET/app/harness.py"

echo ""
echo "Verification:"
grep -n "def on_episode_reward" "$TARGET/app/neural/pipeline.py" || echo "  FAIL: on_episode_reward"
grep -n "struct.pack"           "$TARGET/app/neural/sidecar.py"  || echo "  FAIL: struct.pack"
grep -n "threading.local"       "$TARGET/app/neural/synaptic.py" || echo "  FAIL: threading.local"
grep -n "def _step_coactivation" "$TARGET/app/harness.py"        || echo "  FAIL: _step_coactivation"

echo ""
echo "Done. Review with: git -C $TARGET status && git -C $TARGET diff --stat"
echo "Then: git -C $TARGET add app/ && git -C $TARGET commit -m 'v2.3: hardening + per-step co-activation + live MemRL reward'"
