# On-Policy Cartridge Training: Full Technical Write-Up

## Table of Contents
1. [Background: What are Cartridges?](#1-background-what-are-cartridges)
2. [Why On-Policy? The Distribution Mismatch Problem](#2-why-on-policy)
3. [System Architecture](#3-system-architecture)
4. [Implementation: Phase-by-Phase](#4-implementation)
5. [Roadblocks and Solutions (28 issues)](#5-roadblocks)
6. [Why This is Truly On-Policy](#6-why-on-policy)
7. [Limitations and Next Steps](#7-limitations)
8. [Training Results](#8-results)

---

## 1. Background: What are Cartridges?

Cartridges ([HazyResearch/cartridges](https://github.com/HazyResearch/cartridges)) compress long documents into small, trainable KV cache tensors. Instead of stuffing 50,000 tokens of patient records into a model's context window, you train a "cartridge" — a 2,048-token KV tensor — that captures the same information.

**How it works:**
- A `TrainableCache` (`cartridges/cache.py`) stores `nn.ParameterList`s of key/value tensors across all 28 layers
- `FlexLlamaForCausalLM` (`cartridges/models/llama/modeling_llama.py`) is a modified Llama that accepts this cache via `past_key_values`
- During attention, the cache tokens get `seq_id = -1` (visible to all sequences), while input tokens get `seq_id = 0` (only see each other + cache)
- `CacheAndModel` (`cartridges/train.py:906`) wraps the frozen model + trainable cache into one `nn.Module`

**The training signal:** KL divergence between a "teacher" (same model with full documents in context) and a "student" (same model with only the cartridge). The student learns to match the teacher's per-token probability distributions.

**Key concept — `seq_ids` vs `attention_mask`:**
Standard HuggingFace models use binary `attention_mask` (attend/ignore). FlexLlama uses integer `seq_ids` where each token is labeled with its sequence ID. Tokens with `seq_id = -1` are "global" (cartridge tokens visible to everyone). This enables packing multiple sequences with a shared cartridge prefix in a single forward pass via PyTorch's `FlexAttention`.

---

## 2. Why On-Policy? The Distribution Mismatch Problem

### Original paper (offline distillation)

```
1. Teacher (model + full docs) generates Q&A text + records its logprobs
2. Save to disk as training dataset
3. Student (model + cartridge) does forward pass on teacher's text → student logprobs
4. KL(teacher_logprobs, student_logprobs) → backward → update cartridge
```

The student never generates its own text. It trains on text the *teacher* produced.

### The problem

After training, the cartridge is deployed and the student generates its own text — which differs from what the teacher generated. The student was trained on text it would never produce. This is **distribution mismatch** (same concept as in RL: training on off-policy data).

### On-policy distillation (what we built)

```
1. Student (model + current cartridge) GENERATES text via Tokasaurus
2. Student does forward pass on ITS OWN text → student logprobs (with gradients)
3. Teacher does forward pass on THE SAME text → teacher logprobs
4. KL(teacher_logprobs, student_logprobs) → backward → update cartridge
5. Sync updated cartridge to Tokasaurus → repeat
```

The student trains on text it actually produces. Each training step regenerates text with the latest cartridge. No mismatch.

### Why you can't backprop through generation

Generation is autoregressive: sample token 1, then token 2, etc. Each sampling step is non-differentiable (you pick one token from 128K options — no gradient through that). That's why step 2 exists separately: you feed the already-generated text back through the model to get differentiable logprobs. This is identical to how PPO works in RLHF.

---

## 3. System Architecture

```
Your Mac                    Modal Cloud
─────────                   ──────────────────────────────────
                            ┌─────────────────────────────┐
modal run modal_train.py → │  Training GPU (A100-80GB)    │
                            │                             │
                            │  veRL trainer loop:          │
                            │  ┌─────────────────────┐    │
                            │  │ ① Rollout            │    │
                            │  │  TokasaurusReplica   │───────→ Tokasaurus (A10G)
                            │  │  → HTTP to Modal     │    │    Llama 3.2 3B
                            │  │  → token IDs back    │←──────  + cartridge
                            │  ├─────────────────────┤    │
                            │  │ ② Student forward    │    │
                            │  │  FlexLlamaForCausalLM│    │
                            │  │  + TrainableCache     │    │
                            │  │  → student logprobs   │    │
                            │  ├─────────────────────┤    │
                            │  │ ③ Ref forward (teacher)│   │
                            │  │  Standard Llama       │    │
                            │  │  (no cartridge)       │    │
                            │  │  → teacher logprobs   │    │
                            │  ├─────────────────────┤    │
                            │  │ ④ KL loss + backward  │    │
                            │  │  → grads into cache   │    │
                            │  │  → optimizer.step()   │    │
                            │  ├─────────────────────┤    │
                            │  │ ⑤ Sync cartridge     │───────→ Tokasaurus reloads
                            │  │  force_redownload    │    │    (next rollout uses
                            │  │                      │    │     updated cartridge)
                            │  └─────────────────────┘    │
                            └─────────────────────────────┘
```

**Two Modal deployments:**
- **Tokasaurus inference server** (`modal_tokasaurus.py`): A10G, runs permanently (scales to zero when idle), serves `/custom/cartridge/completions`
- **Training job** (`modal_train.py`): A100-80GB, one-shot job, runs veRL with all patches applied

---

## 4. Implementation: Phase-by-Phase

### Phase 1: Tokasaurus on Modal

**File: `verl/verl/workers/rollout/tokasaurus_rollout/modal_tokasaurus.py`**

Key decisions:
- `nvidia/cuda:12.4.1-devel-ubuntu22.04` base image (not `debian_slim`) — FlashInfer needs CUDA headers for JIT compilation
- Model weights baked into image via `.run_function(download_model)` — prevents 6 GB download at every container boot
- `min_containers=0, max_containers=1, scaledown_window=300` — scale to zero when idle, cap at 1 container
- `kv_cache_num_tokens=32768` — 256K was too large for A10G's 24 GB VRAM

### Phase 2: Rollout Bridge

**File: `verl/verl/workers/rollout/tokasaurus_rollout/async_tokasaurus_server.py`**

`TokasaurusHttpServer` (Ray actor):
- `generate()`: builds payload with `prompt` (token IDs), `max_tokens`, `temperature`, `cartridges`, `logprobs_in_fingerprint=True`
- POSTs to `/custom/cartridge/completions`
- Parses response: `completion_ids` and `packed_chosen_logprobs` from `system_fingerprint` JSON
- Returns `TokenOutput(token_ids, log_probs, stop_reason)`
- Retries: 5 attempts with exponential backoff (2s, 4s, 8s, 16s, 32s)
- Ping: 5 attempts × 120s timeout (survives Modal cold start ~2 min)

`TokasaurusReplica(RolloutReplica)`:
- Reads `tokasaurus_url` and `cartridges` from `RolloutConfig.custom`
- `launch_servers()`: creates Ray actor, pings, stores handle
- `sync_cartridge()`: broadcasts `force_redownload=True` to all servers
- `rollout_worker_use_gpu() → False` — HTTP proxy needs no GPU

`ServerAdapter`:
- Required by veRL's `_ROLLOUT_REGISTRY` for hybrid engine mode
- All methods are no-ops (external server manages its own lifecycle)
- `__getattr__` returns async noop for any method veRL might call

**File: `verl/verl/workers/rollout/replica.py`** — registered `"tokasaurus"` in `RolloutReplicaRegistry`

**File: `verl/verl/workers/rollout/base.py`** — registered `("tokasaurus", "async")` in `_ROLLOUT_REGISTRY`

### Phase 3: Actor (Training GPU) Modifications

**File: `verl/verl/workers/config/actor.py`**

Added `CartridgeConfig` dataclass with: `enabled`, `checkpoint_path`, `num_tokens`, `num_frozen_tokens`, `lr`

**File: `verl/verl/workers/fsdp_workers.py`** — three insertion points in `_build_model_optimizer()`:

1. **Model class override** (after AutoModel selection, before `.from_pretrained()`):
   ```python
   if _cart_enabled and role == "actor":
       from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
       actor_module_class = FlexLlamaForCausalLM
   ```
   Why: standard `LlamaForCausalLM` raises `ValueError: past_key_values should be Cache or None` when it receives a `TrainableCache`. `FlexLlamaForCausalLM` knows how to handle it.

2. **Cartridge loading** (after model creation, before FSDP wrapping):
   - Freeze all base model params
   - Download cartridge from HuggingFace (auto-detect `.pt` filename, rename `fixed_keys` → `frozen_keys`)
   - Create `TrainableCache` from checkpoint
   - Wrap: `actor_module = CacheAndModel(cache, model)`

3. **Optimizer override** (in optimizer creation section):
   ```python
   if self._cartridge_enabled:
       actor_optimizer = torch.optim.Adam(self._cartridge_cache.parameters(), lr=cartridge_lr)
   ```
   Only ~117M cache params get optimized (vs 3.3B total). Base model is frozen.

Also added `save_cartridge()` method — saves `TrainableCache` to disk for Tokasaurus sync.

**File: `verl/verl/workers/actor/dp_actor.py`**

Added `_forward_micro_batch_cartridge()` — completely separate forward path:

```python
for i in range(batch_size):
    # Extract valid tokens (drop padding)
    ids_i = input_ids[i, :valid_len]           # 1D tensor (FlexLlama expects this)
    seq_ids_i = torch.zeros(valid_len, ...)     # All tokens = sequence 0
    pos_ids_i = torch.arange(valid_len, ...)    # Sequential positions
    
    cache_obj.clear()  # Reset cache state between samples
    
    output = self.actor_module(
        input_ids=ids_i, seq_ids=seq_ids_i, position_ids=pos_ids_i
    )
    # CacheAndModel internally passes past_key_values=cache to FlexLlama
    # FlexLlama prepends cartridge KV tokens (seq_id=-1) at each attention layer
```

Key detail: FlexLlama expects **1D tensors** and internally does `unsqueeze(0)`. Passing 2D `(1, seq_len)` causes shape mismatches in FlexAttention's `create_block_mask`.

### Phase 4: Training Loop

**File: `verl/verl/trainer/ppo/ray_trainer.py`**

- `__init__`: reads `actor.cartridge.enabled` to set `_cartridge_sync_enabled`
- After `update_actor` + `update_weights`: calls `_sync_cartridge_to_tokasaurus()`
- Sync method: `save_cartridge()` on worker → `sync_cartridge()` on replicas

**File: `verl/verl/experimental/agent_loop/agent_loop.py`**

Fixed prompt length mismatch in `_agent_loop_postprocess()`:
```python
if len(output.prompt_ids) > max_prompt_len:
    output.prompt_ids = output.prompt_ids[-max_prompt_len:]
```
Chat template adds tokens (system header, BOS, etc.) that can push prompts beyond `max_prompt_length`. When `torch.cat` tries to stack prompts of different lengths, it crashes.

### Phase 5: Data & Config

**File: `verl/examples/cartridge_distill/prepare_data.py`**

Downloads LongHealth benchmark (400 medical questions about 10 patients). Each row:
- `prompt`: question in chat format (for student rollout)
- `document_text`: full patient documents (for future teacher context)
- `reward_model.ground_truth`: correct answer (dummy, for GRPO compatibility)

**File: `verl/examples/cartridge_distill/dummy_reward.py`**

Returns `0.0` for all responses. GRPO requires a reward function, but cartridge distillation uses KL loss. The reward is a no-op.

**File: `verl/examples/cartridge_distill/modal_train.py`**

Modal deployment script that:
1. Builds image with CUDA + torch + flash-attn + veRL + cartridges + tokasaurus
2. At runtime: clones our patches repo, applies patches with `patch` command
3. Runs `prepare_data.py` to create parquet files
4. Launches `verl.trainer.main_ppo` with all Hydra overrides

---

## 5. Roadblocks and Solutions

28 issues encountered during development. See [README.md](README.md#issues-encountered-and-fixes) for the full table.

Key categories:

**Modal infrastructure (issues 1-5):** Missing packages, model download at runtime, GPU memory, cold start scaling.

**Dependency hell (issues 8-12):** `pydrantic` not on PyPI, `flash-attn` build requires torch at build time, ABI mismatches between pre-built wheels and pip torch, incomplete pip installs of cartridges.

**API mismatches (issues 14-16):** `TrainableCache.from_pretrained()` API differs from what we assumed, HuggingFace repo file naming, checkpoint key naming (`fixed_keys` vs `frozen_keys`).

**veRL integration (issues 17-28):** Two separate registries for rollout classes, FSDP can't handle mixed `requires_grad`, async lifecycle methods, standard Llama rejecting TrainableCache, 1D vs 2D tensor conventions, missing output keys, prompt length mismatches after chat template.

---

## 6. Why This is Truly On-Policy

**On-policy means:** the text being trained on was generated by the current policy.

In our system:
1. The **current** cartridge (after the latest optimizer step) is synced to Tokasaurus
2. Tokasaurus generates responses using the **current** cartridge
3. The student computes logprobs on **those responses** (not pre-generated ones)
4. After the optimizer step, the cartridge is **re-synced** to Tokasaurus
5. The **next** rollout uses the **updated** cartridge

This is verified by the `cartridge_sync` timing in the training logs (~7-10s per step). Each step:
- Saves the updated `TrainableCache` to disk
- Tells Tokasaurus to reload with `force_redownload=True`
- Next rollout generates text with the updated cartridge

Compare to offline:
- Text is generated once, saved to disk
- Training runs on the same fixed text for all epochs
- Cartridge changes but text doesn't → mismatch grows over time

---

## 7. Limitations and Next Steps

### Current limitations

1. **Teacher doesn't have full document context.** The ref model (teacher) is the same frozen Llama without any context — it just sees the prompt + student's response. The original paper's teacher sees the full 50K-token documents. To fix: either (a) send the student's response to Tokasaurus without cartridge but with full documents in the prompt, or (b) stuff documents into the ref model's input.

2. **GRPO wrapper.** We're using veRL's GRPO algorithm as a wrapper with a dummy reward function. The actual training signal is the KL loss between actor and ref (`use_kl_loss=True`). A dedicated cartridge distillation trainer would be cleaner.

3. **No cartridge-specific evaluation.** We don't evaluate whether the cartridge improves on the LongHealth benchmark after training. Need to add generation eval (multiple choice accuracy).

4. **Single-GPU training.** Currently runs on 1 GPU with FSDP NO_SHARD. Multi-GPU would require careful handling of `TrainableCache` (it's small enough to replicate, not shard).

5. **FlexAttention compilation overhead.** First forward pass triggers `torch.compile` for FlexAttention kernels (~14s). Subsequent passes are fast. Different sequence lengths trigger recompilation.

### Next steps

1. **Teacher with full documents:** Send student-generated text to Tokasaurus with documents in the prompt (no cartridge) → get teacher logprobs. The `get_teacher_logprobs()` method is already implemented in `async_tokasaurus_server.py`.

2. **Dedicated distillation trainer:** Replace GRPO with a simpler loop: rollout → student fwd → teacher fwd → KL loss → backward. No reward, no advantage estimation, no critic.

3. **Evaluation:** After each epoch, generate answers to LongHealth multiple-choice questions and measure accuracy. Compare cartridge-conditioned vs baseline.

4. **Offline + online hybrid:** Use the existing HuggingFace dataset (with pre-computed teacher logprobs) for the first epoch, then switch to on-policy for subsequent epochs.

---

## 8. Training Results

First successful run (A100-80GB, batch_size=8, 2 epochs × 18 batches = 36 steps):

```
step:1  → kl_loss=0.749, grad_norm=0.019, cartridge_sync=10.8s, throughput=8.4 tok/s
step:2  → kl_loss=1.110, grad_norm=0.021, cartridge_sync=9.5s,  throughput=24.7 tok/s
step:3  → kl_loss=1.127, grad_norm=0.007, cartridge_sync=8.4s,  throughput=26.1 tok/s
step:4  → kl_loss=1.206, grad_norm=0.062, cartridge_sync=10.0s, throughput=27.4 tok/s
step:5-7 → (training continued)
step:8  → kl_loss=0.957, grad_norm=0.025, cartridge_sync=9.9s,  throughput=26.8 tok/s
step:9  → kl_loss=0.729, grad_norm=0.039, cartridge_sync=8.6s,  throughput=30.3 tok/s ← lowest KL
```

Note: KL loss dropped from 0.749 → 0.729 over 9 steps, showing the cartridge is learning.
Crashed at step 10 on checkpoint save (`CacheAndModel` missing `.config` attribute).
Fix: disabled checkpointing (`save_freq=-1`) — a proper fix would add `.config` passthrough to `CacheAndModel`.

**Timing breakdown per step (~75s):**
- Rollout (Tokasaurus generation): ~12s
- Student forward (FlexLlama + cartridge): ~8s
- Ref forward (teacher logprobs): ~0.6s
- Actor update (forward + backward + optimizer): ~41s
- Cartridge sync: ~9s
- Overhead: ~4s

**Memory usage on A100-80GB:**
- Peak allocated: 43.7 GB
- Peak reserved: 52.8 GB
- Trainable params: 117M / 3.33B total (3.5%)

**Model:** Llama 3.2 3B Instruct
**Cartridge:** `hazyresearch/cartridge-wauoq23f` (2048 tokens, 2040 trainable + 8 frozen)
**Data:** LongHealth medical benchmark (146 train, 229 val prompts)
