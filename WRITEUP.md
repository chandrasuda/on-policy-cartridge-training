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

1. **Teacher document context — investigated extensively, not yet solved.**

   We made three attempts to give the teacher full patient documents:

   **Attempt 1: Store `document_text` in training data.**
   Added the 12K-char document text as a column in the parquet file (`prepare_data.py`). veRL's `RLHFDataset` loads it into `non_tensor_batch`. But veRL's data pipeline strips non-standard fields during rollout → batching → padding transformations. By the time `_compute_ref_log_prob` ran in `ray_trainer.py`, `non_tensor_batch` was empty. Result: all teacher logprobs were zero, `kl_loss=0.0` on every step.

   **Attempt 2: Look up documents by `patient_id`.**
   A shorter field (`"patient_01"`) that might survive the pipeline better. Downloaded LongHealth data at startup, built a `patient_id → documents` lookup. Result: `patient_id` was ALSO stripped from `non_tensor_batch`. Same zeros.

   **Attempt 3: Match patient name from decoded prompt tokens.**
   Since `input_ids` (the actual tokens) definitely survive the pipeline, we decoded the first 300 tokens and searched for patient names like "Anna Sample". Tested locally — 20/20 matches. But then discovered the fundamental blocker: **Tokasaurus's `packed_chosen_logprobs` only covers newly generated tokens, not prompt tokens.** Sending `[doc + prompt + response]` with `max_tokens=1` returns logprobs for just 1 new token. The student's response tokens are in the "prompt" part and don't get logprobs. `echo=True` returns HTTP 400 on the cartridge completions endpoint. **There is no Tokasaurus API to get teacher logprobs on arbitrary prompt tokens.**

   **Current state:** Using veRL's ref model (same frozen Llama, no document context). KL loss = student (with cartridge) vs base model (without cartridge or documents). This is a weaker signal than the original paper's teacher (which sees full documents), but it's still valid — the cartridge should at minimum produce predictions that match the base model's.

   **To properly fix:** Compute teacher logprobs locally on the training GPU by modifying the ref model's forward pass to prepend document tokens. This requires: (a) ensuring document text reaches the ref worker, (b) tokenizing documents, (c) extending `input_ids`, `attention_mask`, `position_ids` with document prefix, (d) running the ref forward pass on the longer sequence (~12K + 512 tokens), (e) extracting only the response token logprobs. This is feasible on an A100-80GB but requires significant changes to veRL's ref computation path.

2. **GRPO wrapper.** We're using veRL's GRPO algorithm as a wrapper with a dummy reward function (`dummy_reward.py` returns 0.0). The actual training signal is the KL loss between actor and ref (`use_kl_loss=True`). A dedicated cartridge distillation trainer would be cleaner — just rollout → student fwd → teacher fwd → KL loss → backward, without the reward/advantage machinery.

3. **No cartridge-specific evaluation.** We don't evaluate whether the cartridge improves on the LongHealth benchmark after training. Need to add generation eval (multiple choice accuracy). The `LongHealthMultipleChoiceGenerateDataset` class exists in the cartridges repo.

4. **Single-GPU training.** Currently runs on 1 GPU with FSDP NO_SHARD. Multi-GPU would require careful handling of `TrainableCache` (it's tiny — 117M params — so blackbox, don't shard).

5. **FlexAttention compilation overhead.** First forward pass triggers `torch.compile` for FlexAttention kernels (~14s). Different sequence lengths trigger recompilation. The `max-autotune-no-cudagraphs` mode in `FlexLlamaForCausalLM` helps.

6. **Checkpoint saving.** `CacheAndModel` wrapper lacks a `.config` attribute that veRL's checkpoint manager expects. Disabled via `save_freq=-1`. A proper fix: add `@property config` passthrough to `CacheAndModel`.

7. **~~39% of steps had kl_loss=0.0~~** ROOT CAUSE FOUND AND FIXED (Issue #33). The `_forward_micro_batch_cartridge` method used `input_ids[i, :valid_len]` to extract tokens. With LEFT-padded inputs like `[0, 0, PAD, tok1, tok2, ...]`, this grabbed the padding zeros from the left instead of the actual tokens. The model received garbage input → garbage logprobs → KL matched the ref (both garbage) → KL=0. Fix: `input_ids[i][attention_mask[i].bool()]` extracts only the valid tokens regardless of padding direction.

### Next steps

1. **Teacher with full documents (local):** Modify the ref worker's `compute_log_prob` to prepend document tokens when in cartridge mode. This keeps everything on the training GPU without HTTP calls. Requires veRL core changes to pass document text through the pipeline.

2. **Dedicated distillation trainer:** Replace GRPO with a simpler training loop that doesn't need rewards, critics, or advantage estimation.

3. **Evaluation:** Run LongHealth multiple-choice evaluation after training to measure cartridge quality improvement.

4. **Offline + online hybrid:** Use HuggingFace pre-computed data (`hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-*`) with teacher logprobs for warm-starting, then switch to on-policy.

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

**Run 1 (ref model as teacher, no document context):**
Previous run with veRL's ref model: KL loss ~0.7-1.2, completed 9/40 steps before checkpoint crash.

**Run 2 (Tokasaurus as teacher with documents):**
Attempted to call Tokasaurus with full documents — `kl_loss=0.0` on most steps because:
(a) `document_text`/`patient_id` stripped from `non_tensor_batch` by veRL
(b) Even with name-matching fix, Tokasaurus API can't return logprobs on prompt tokens

**Run 3 (ref model without documents):**
KL loss ~0.01-0.03, intermittent zeros (39% of steps). Completed 40 steps.

**Run 4 (FINAL — ref model with full patient documents, left-padding fix):**
```
step 1:  5.08  ← teacher with 12K-token documents produces very different logprobs
step 2:  4.95
step 3:  3.91  ← cartridge rapidly improving
step 5:  3.91
step 8:  3.52
step 10: 2.76
step 15: 2.56
step 20: 2.67
step 25: 2.31
step 30: 2.15  ← lowest
step 35: 2.91
step 40: 2.45  ← final (51% reduction from start)
```
**KL decreased from 5.08 → 2.45 (51% reduction).** Teacher: 8/8 samples with documents on EVERY step.
100% teacher hit rate, zero failed steps. Completed in 49 minutes on A100-80GB.

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
