# v2.3 Hardening Patch

Apply this to `ljrseals-dev/pinecone-neural` (or any checkout of the pre-hardening neural tree).

## What this patch does

| File | Change |
|------|--------|
| `app/neural/pipeline.py` | Adds `on_episode_reward()` (wires reward → UtilityModel) |
| `app/neural/sidecar.py` | Binary float32 via `struct` + `BLOB` (replaces JSON) |
| `app/neural/synaptic.py` | `threading.local()` connections, `check_same_thread=True` |
| `app/neural/utility.py` | Full MemRL-style MLP (create if missing) |
| `app/neural/__init__.py` | Exports Layer 4 symbols |
| `app/harness.py` | Per-step `_step_coactivation()` + live reward call |

## Apply

```bash
# From the root of your clone of pinecone-neural
git clone https://github.com/ljrseals-dev/pinecone-neural
cd pinecone-neural

# Copy the patched files
cp /path/to/v2.3-patch/pipeline.py      app/neural/pipeline.py
cp /path/to/v2.3-patch/sidecar.py       app/neural/sidecar.py
cp /path/to/v2.3-patch/synaptic.py      app/neural/synaptic.py
cp /path/to/v2.3-patch/utility.py       app/neural/utility.py
cp /path/to/v2.3-patch/neural_init.py   app/neural/__init__.py
cp /path/to/v2.3-patch/harness.py       app/harness.py

# Verify
grep -n "def on_episode_reward" app/neural/pipeline.py
grep -n "struct.pack" app/neural/sidecar.py
grep -n "threading.local" app/neural/synaptic.py
grep -n "def _step_coactivation" app/harness.py

# Commit
git add app/neural/ app/harness.py
git commit -m "v2.3: binary sidecar, thread-safe synaptic, per-step co-activation, live MemRL reward"
git push
```

## One-liner (from this zip)

```bash
cd pinecone-neural && \
cp ../v2.3-patch/pipeline.py app/neural/ && \
cp ../v2.3-patch/sidecar.py app/neural/ && \
cp ../v2.3-patch/synaptic.py app/neural/ && \
cp ../v2.3-patch/utility.py app/neural/ && \
cp ../v2.3-patch/neural_init.py app/neural/__init__.py && \
cp ../v2.3-patch/harness.py app/ && \
git add app/ && git status
```

## After apply — expected grep results

```
app/neural/pipeline.py:61:    def on_episode_reward(
app/neural/sidecar.py:55:        return struct.pack(f"{len(vector)}f", *vector)
app/neural/synaptic.py:37:        self._local = threading.local()
app/harness.py:83:    def _step_coactivation(self, step_ids: List[str], success: bool) -> None:
```

If any of those are empty, the copy did not land.
