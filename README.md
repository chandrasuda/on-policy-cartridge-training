# On-Policy Cartridge Training

On-policy Cartridge distillation via **Tokasaurus** + **veRL**.

The original [Cartridges paper](https://github.com/HazyResearch/cartridges) trains KV cache tensors using offline distillation — the teacher pre-generates text and logprobs, then the student trains on that fixed data. This project enables **on-policy** distillation where the student generates text live through Tokasaurus, then both student and teacher compute logprobs on that text, eliminating distribution mismatch.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              On-Policy Cartridge Distillation Loop            │
│                                                               │
│  ① STUDENT ROLLOUT (Tokasaurus on Modal, no gradients)       │
│     Student = frozen_model + current_cartridge → generate()   │
│     Output: response token IDs                                │
│                                                               │
│  ② STUDENT FORWARD PASS (veRL training GPU, WITH gradients)  │
│     Student = frozen_model + cartridge → forward(response)    │
│     Output: differentiable student_logprobs                   │
│                                                               │
│  ③ TEACHER FORWARD PASS (same model + full document context) │
│     Teacher = frozen_model + FULL DOCS → forward(response)    │
│     Output: teacher_logprobs                                  │
│                                                               │
│  ④ KL DISTILLATION UPDATE                                    │
│     loss = KL(teacher_logprobs, student_logprobs)             │
│     loss.backward() → gradients into cartridge params only    │
│     optimizer.step() → cartridge updated                      │
│                                                               │
│  ⑤ SYNC CARTRIDGE → Tokasaurus reloads for next rollout      │
└─────────────────────────────────────────────────────────────┘
```

## What's in this repo

### New files (drop into veRL)

| File | Purpose |
|------|---------|
| `verl_patches/rollout/tokasaurus_rollout/` | Rollout bridge: `TokasaurusHttpServer` + `TokasaurusReplica` |
| `modal/modal_tokasaurus.py` | Modal deployment for Tokasaurus inference server |
| `training/run_cartridge_distill.sh` | Training config that wires everything together |
| `training/prepare_data.py` | Downloads LongHealth data → veRL parquet format |
| `tests/` | Unit tests + integration tests |

### Patches to veRL (apply with `git apply`)

| Patch | What it changes |
|-------|----------------|
| `verl_patches/config/actor_config.patch` | Adds `CartridgeConfig` to `ActorConfig` |
| `verl_patches/fsdp_workers.patch` | Loads `TrainableCache`, freezes model, wraps with `CacheAndModel` |
| `verl_patches/actor/dp_actor.patch` | Cartridge-specific forward pass (`seq_ids` instead of `attention_mask`) |
| `verl_patches/rollout/replica.patch` | Registers `"tokasaurus"` in `RolloutReplicaRegistry` |

## Setup

### 1. Prerequisites

```bash
# Clone veRL and cartridges
git clone https://github.com/volcengine/verl.git
git clone https://github.com/HazyResearch/cartridges.git
git clone https://github.com/chandrasuda/tokasaurus.git -b geoff/cartridges

pip install -e verl/
pip install -e cartridges/
```

### 2. Apply patches to veRL

```bash
cd verl/
git apply ../on-policy-cartridge-training/verl_patches/config/actor_config.patch
git apply ../on-policy-cartridge-training/verl_patches/fsdp_workers.patch
git apply ../on-policy-cartridge-training/verl_patches/actor/dp_actor.patch
git apply ../on-policy-cartridge-training/verl_patches/rollout/replica.patch

# Copy new files
cp -r ../on-policy-cartridge-training/verl_patches/rollout/tokasaurus_rollout/ \
      verl/workers/rollout/tokasaurus_rollout/
```

### 3. Deploy Tokasaurus on Modal

```bash
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN

modal deploy on-policy-cartridge-training/modal/modal_tokasaurus.py
# → gives you a URL like https://YOUR_USER--tokasaurus-cartridge-server-serve.modal.run
```

### 4. Test the rollout

```bash
# Quick test
python on-policy-cartridge-training/verl_patches/rollout/tokasaurus_rollout/example_query.py

# Full veRL rollout test (uses Ray)
python on-policy-cartridge-training/verl_patches/rollout/tokasaurus_rollout/test_rollout.py
```

### 5. Prepare data & train

```bash
# Download LongHealth → parquet
python on-policy-cartridge-training/training/prepare_data.py

# Launch training (needs GPU)
TOKASAURUS_URL=https://YOUR_URL bash on-policy-cartridge-training/training/run_cartridge_distill.sh
```

## Key design decisions

- **Tokasaurus runs externally on Modal** — the rollout bridge is just an HTTP proxy (Ray actor). No GPU needed for the proxy.
- **Cartridge is NOT wrapped in FSDP** — it's tiny (~50M params for 2048 tokens). Only the frozen base model gets FSDP.
- **Optimizer targets `cache.parameters()` only** — base model is frozen, only cartridge KV tensors receive gradients.
- **`FlexLlamaForCausalLM`** uses `seq_ids` instead of `attention_mask` — enables packed sequences with shared cartridge prefix via FlexAttention.
- **Cartridge sync via `force_redownload`** — after each optimizer step, save the cartridge and tell Tokasaurus to reload it.

## Status

- ✅ Tokasaurus Modal deployment (working, tested)
- ✅ Rollout bridge: TokasaurusReplica + TokasaurusHttpServer
- ✅ Cartridge injection verified (base model vs cartridge-conditioned)
- ✅ Actor: CartridgeConfig, TrainableCache loading, CacheAndModel wrapping
- ✅ Actor: cartridge-specific forward pass
- ✅ Optimizer targeting cache.parameters() only
- ✅ Training config + data pipeline
- 🔲 Teacher forward pass with full document context (currently uses veRL's ref model)
- 🔲 End-to-end training run on GPU

## See also

- [PLAN.md](PLAN.md) — detailed implementation plan
- [Cartridges paper](https://github.com/HazyResearch/cartridges)
- [Tokasaurus](https://github.com/chandrasuda/tokasaurus/tree/geoff/cartridges)
- [veRL](https://github.com/volcengine/verl)
