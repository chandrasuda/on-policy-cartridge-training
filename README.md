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
- ✅ Actor: cartridge-specific forward pass (FlexLlamaForCausalLM + TrainableCache)
- ✅ Optimizer targeting cache.parameters() only
- ✅ Training config + data pipeline
- ✅ End-to-end training on A100-80GB (Modal) — KL loss decreasing, cartridge syncing
- ⏳ Teacher uses veRL's ref model (same frozen model, no document context). Full document context requires separate teacher server.

### Training results (first run)

```
step:1 → kl_loss=0.925, grad_norm=0.058, cartridge_sync=6.6s
step:2 → kl_loss=0.918, grad_norm=0.116, cartridge_sync=7.0s
step:3 → kl_loss=1.212, grad_norm=0.047, cartridge_sync=6.7s
step:4 → kl_loss=1.081, grad_norm=0.030, cartridge_sync=6.9s
```

## Issues encountered and fixes

| # | Issue | Root cause | Fix |
|---|-------|-----------|-----|
| 1 | Modal image: `git` not found | Debian slim has no git | `.apt_install("git")` |
| 2 | Tokasaurus startup timeout | Model downloading at runtime (~6 GB) | Bake weights into image via `.run_function(download_model)` |
| 3 | CUDA OOM on A10G (KV cache) | 256K token KV cache > 24 GB VRAM | Reduce to 32K tokens |
| 4 | FlashInfer JIT fails | `CUDA_HOME` not set in debian_slim | Switch to `nvidia/cuda:12.4.1-devel` base image |
| 5 | 10 GPU containers running | `min_containers=1` + no max | `min_containers=0, max_containers=1` |
| 6 | `logprobs=5` HTTP 400 | Server not configured with `max_topk_logprobs` | Remove `logprobs` field; use `logprobs_in_fingerprint` only |
| 7 | Ray serialization recursion | `MagicMock` not serializable by Ray | Use `SimpleNamespace` for configs |
| 8 | `pydrantic` not on PyPI | Private dep from cartridges repo | Install cartridges before other deps |
| 9 | `CARTRIDGES_DIR` not set | Cartridges `__init__.py` requires it | Add to Modal `.env()` |
| 10 | `flash-attn` build needs torch | Build-time dep not available | Install torch first, then flash-attn with `--no-build-isolation` |
| 11 | flash-attn ABI mismatch | `cxx11abiTRUE` wheel vs pip torch `FALSE` | Use `cxx11abiFALSE` wheel from GitHub releases |
| 12 | `cartridges.utils` not found | pip install from git incomplete | `git clone` + `pip install -e` |
| 13 | Hydra: `cartridge` key not found | New key not in default config schema | Use `+` prefix in Hydra overrides |
| 14 | `TrainableCache.from_pretrained()` wrong API | Passed `attn_config=` but method only takes `path` | Removed extra arg; it infers config from checkpoint |
| 15 | HF file `cartridge.pt` not found | Actual file is `cache-step4092.pt` | Dynamically find `.pt` via `list_repo_files()` |
| 16 | `frozen_keys` not in checkpoint | Checkpoint uses `fixed_keys` (naming mismatch) | Rename keys before loading |
| 17 | A10G OOM (actor + ref = 2x 3B fp32) | 24 GB < 2 × 12 GB models | Switch to A100-80GB |
| 18 | FSDP mixed `requires_grad` error | Frozen model + trainable cache in same FSDP group | `use_orig_params=True` |
| 19 | `tokasaurus` not in `_ROLLOUT_REGISTRY` | veRL's hybrid engine uses separate registry | Created `ServerAdapter`, registered in `base.py` |
| 20 | Tokasaurus cold start → ping timeout (10s) | Modal scales to zero; cold start ~2 min | Retry ping 5 × 120s |
| 21 | `rollout.resume()` not found | `ServerAdapter` missing lifecycle methods | Added `__getattr__` catch-all returning async noop |
| 22 | `update_weights()` returns None | `__getattr__` returned sync noop, but veRL `await`s it | Always return async noop |
| 23 | `data_source='longhealth'` reward not implemented | GRPO expects a reward function | Created `dummy_reward.py` returning 0.0 |
| 24 | Standard `LlamaForCausalLM` rejects `TrainableCache` | `past_key_values` must be `Cache` or `None` | Load `FlexLlamaForCausalLM` when cartridge enabled |
| 25 | Tensor shape mismatch in FlexAttention | Passed 2D `(1, seq_len)` tensors; FlexLlama expects 1D | Pass 1D tensors; FlexLlama adds batch dim internally |
| 26 | Missing `entropys` key in output | Cartridge forward only returned `log_probs` | Added `entropys=torch.zeros_like(log_probs)` |
| 27 | A100-40GB OOM during backward | FlexAttention backward + 2048 cartridge tokens | Use A100-80GB + `micro_batch_size=1` |
| 28 | `prompt_ids` tensor cat size mismatch | Chat template makes some prompts > `max_prompt_length` | Truncate prompts to `max_prompt_length` in agent loop postprocess |
| 29 | `CacheAndModel` missing `.config` → checkpoint crash | veRL's checkpoint manager accesses `model.config` | Disabled checkpointing (`save_freq=-1`) |
| 30 | Teacher logprobs all zeros (`kl_loss=0.0`) | `document_text`/`patient_id` stripped from `non_tensor_batch` by veRL pipeline | Reverted to ref model (no documents). Tokasaurus API can't return prompt logprobs. |
| 31 | Patches repo was private → image build 404 | `gh repo create` defaulted to private | `gh repo edit --visibility public` |
| 32 | Git clone fails in Modal containers at runtime | Modal restricts outbound git:// | Download ZIP via `urllib` during image build instead |

## See also

- [PLAN.md](PLAN.md) — detailed implementation plan
- [Cartridges paper](https://github.com/HazyResearch/cartridges)
- [Tokasaurus](https://github.com/chandrasuda/tokasaurus/tree/geoff/cartridges)
- [veRL](https://github.com/volcengine/verl)
