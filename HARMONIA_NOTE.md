<!--
Copyright (c) 2026 Lewis James Seals Jr.
Author: Lewis James Seals Jr. (@ljrseals-dev)
Project: Pinecone Neural Agent Harness
SPDX-License-Identifier: MIT
-->

# HarmoniaCodec integration note

The HarmoniaCodec-Music package (audited) was inspected.

**Finding:** No trained weights ship with the package. MODEL_STATUS.json
marks all trainable components as untrained / blocked. The README is explicit:
"What does not ship is a trained model."

Therefore:
- Weights cannot be published to Hugging Face until a training run completes
  and the gate attestation passes.
- The codec source remains a separate research scaffold under MIT.
- Integration into this agent harness is deferred until weights exist.

To train and publish later:
1. Follow HarmoniaCodec TRAINING.md / train_synth.py
2. Run harmonia_gate.attest_trained()
3. Push checkpoints to a HF model repo
4. Wire a HarmoniaTool into app/tools/builtin.py that loads the HF weights
