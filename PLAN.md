# Plan: On-Policy Cartridge Distillation via Tokasaurus + veRL

## Goal
Integrate Tokasaurus into veRL to enable **on-policy Cartridge distillation**. The loss function is the same KL divergence as the original Cartridges paper — the key difference is that the text being distilled on is generated **live by the student** (on-policy) rather than pre-generated offline, eliminating the train-serve distribution mismatch.

This requires three systems to work together:
- **Tokasaurus** (geoff/cartridges branch): generates rollouts with Cartridge KV tensor injection (bypasses prefill)
- **veRL**: orchestrates the on-policy distillation training loop
- **Cartridges** (`TrainableCache`): the trainable KV tensor that receives gradients

## On-Policy Distillation (NOT RL/PPO)

### What the original Cartridges paper does (offline distillation)
1. Pre-generate synthetic Q&A offline → save to disk with teacher logprobs
2. Training: forward pass model+cartridge on pre-generated text, compute KL(teacher, student), backward, update cartridge
3. The model never generates text during training

### What we're building (on-policy distillation)
Same loss function (KL), but the text is generated **live**:

```
STEP ① — ROLLOUT (Tokasaurus, inference-only, NO gradients)
   Purpose: "What does the student actually say when using this Cartridge?"
   Student = frozen_model + current_cartridge → autoregressive generation
   Output: response_ids [382, 1029, 553, ...] (just token IDs)
   
STEP ② — STUDENT FORWARD PASS (veRL training GPU, WITH gradients)
   Purpose: "How does the student assign probabilities to those tokens?"
   Student = frozen_model + current_cartridge → forward(prompt + response_ids) → student_logits
   Output: differentiable logits/log_probs (gradients flow into cartridge.trainable_keys/values)
   
STEP ③ — TEACHER FORWARD PASS (replaces the Reward Model)
   Purpose: "How does the teacher assign probabilities to those tokens?"
   Teacher = SAME frozen_model + FULL DOCUMENT (no cartridge) → forward(prompt + response_ids)
   Output: teacher_logprobs (top-k per token, dense per-token signal)

STEP ④ — KL DISTILLATION UPDATE (replaces PPO/GRPO)
   loss = KL(teacher_logprobs, student_logits)    ← same loss as original paper
   loss.backward()       → gradients flow into cartridge.trainable_keys/values
   optimizer.step()      → cache params updated
   
STEP ⑤ — SYNC CARTRIDGE TO TOKASAURUS
   cache.save("/tmp/cartridge.pt")
   Tokasaurus reloads → next rollout uses updated cartridge
```

### Why this is NOT PPO
| | PPO/GRPO | On-Policy Distillation (what we do) |
|---|---|---|
| **Signal** | Sparse reward at end of sequence | Dense teacher logprobs at every token |
| **Reward model** | Needed (separate model or function) | NOT needed |
| **Critic** | Needed for advantage estimation | NOT needed |
| **Loss** | Policy gradient (REINFORCE/PPO clip) | KL divergence (same as original paper) |
| **Teacher** | N/A | Same frozen base model with full document context |

### Key insight: Teacher = same model, different input
The "teacher" is NOT a separate, larger model. It's the **exact same frozen base model** but given the **full raw document** in its context (100k+ tokens) instead of the Cartridge (2048 tokens). The student tries to match the teacher's token-level predictions using only the compressed Cartridge.

### veRL already has on-policy distillation infrastructure
veRL has a documented `OnPolicyDistillTrainer` recipe ([async-on-policy-distill.md](/Users/csuda/cartridges-workspace/verl/docs/advance/async-on-policy-distill.md)) that implements:
- Student rollout → Teacher top-k logprob retrieval → KL loss → Student update
- One-step and two-step off-policy scheduling for throughput
- Megatron actor with KL loss injection
- Teacher served via ZMQ proxy+worker architecture

For Cartridges, we adapt this recipe:
- **Student rollout**: Tokasaurus (with Cartridge injection) instead of vLLM
- **Teacher**: Same base model served with full document context (via separate Tokasaurus/SGLang instance, or same veRL infrastructure)
- **Student update**: `loss.backward()` flows gradients into `TrainableCache` params (not model weights)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│               veRL On-Policy Distillation Loop                     │
│                                                                    │
│  ┌──────────────────┐    ┌─────────────────────────────────────┐   │
│  │ ① STUDENT ROLLOUT│    │ ② STUDENT FORWARD (differentiable) │   │
│  │                  │    │                                     │   │
│  │ Tokasaurus       │    │  CacheAndModel(                    │   │
│  │ + Cartridge      │    │    cache=TrainableCache,            │   │
│  │ = Student policy │    │    model=frozen_base_model           │   │
│  │                  │    │  ).forward(prompt + response_ids)   │   │
│  │ → response_ids   │    │  → student_logits (with gradients)  │   │
│  │   (no gradients) │    │                                     │   │
│  └──────────────────┘    └─────────────────────────────────────┘   │
│                                                                    │
│  ┌──────────────────┐    ┌─────────────────────────────────────┐   │
│  │ ③ TEACHER FWD    │    │ ④ KL DISTILLATION UPDATE            │   │
│  │   (replaces RM)  │    │                                     │   │
│  │                  │    │  loss = KL(teacher_logprobs,         │   │
│  │ Same frozen model│    │           student_logits)            │   │
│  │ + FULL DOCUMENT  │    │  loss.backward()                    │   │
│  │ (no cartridge)   │    │    → grads into cache.trainable_*   │   │
│  │                  │    │  optimizer.step()                    │   │
│  │ → teacher top-k  │    │    → cache params updated           │   │
│  │   logprobs       │    │                                     │   │
│  └──────────────────┘    └─────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ⑤ CARTRIDGE SYNC                                            │   │
│  │  cache.save("/tmp/cartridge.pt") → Tokasaurus reloads       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Phase 1: Rollout Bridge (Steps 1-4)
*"veRL can generate student rollouts through Tokasaurus with Cartridge injection"*

This is the core engineering task. Regardless of whether the downstream math is PPO or KL distillation, veRL needs to be able to generate text through Tokasaurus with a Cartridge injected.

#### Step 1: Create `tokasaurus_rollout/` module
**File**: `verl/verl/workers/rollout/tokasaurus_rollout/__init__.py`
Empty init file.

#### Step 2: Create `TokasaurusHttpServer` Ray actor
**File**: `verl/verl/workers/rollout/tokasaurus_rollout/async_tokasaurus_server.py`

Ray actor that:
- Connects to external Tokasaurus server via HTTP
- Implements `generate(prompt_ids, sampling_params, request_id)` → `TokenOutput`
- Injects `cartridges` array into every request (reads from `RolloutConfig.custom`)
- POSTs to `/custom/cartridge/completions` with raw token IDs
- Extracts `completion_ids` and logprobs from `system_fingerprint` JSON
- Handles retries, timeouts, health checks

Key interface (called by `AsyncLLMServerManager.generate()`):
```python
async def generate(
    self,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    request_id: str,
) -> TokenOutput:
```

#### Step 3: Create `TokasaurusReplica(RolloutReplica)`
**File**: Same file as Step 2

- `launch_servers()`: Creates `TokasaurusHttpServer` Ray actors, verifies Tokasaurus is reachable via `/ping`
- `get_ray_class_with_init_args()`: Returns Ray class for standalone mode
- `wake_up()`, `sleep()`, `clear_kv_cache()`: No-ops (external server)
- Reads Tokasaurus URL and cartridge config from `RolloutConfig.custom`

#### Step 4: Register in `RolloutReplicaRegistry`
**File**: `verl/verl/workers/rollout/replica.py`

```python
def _load_tokasaurus():
    from verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server import TokasaurusReplica
    return TokasaurusReplica

RolloutReplicaRegistry.register("tokasaurus", _load_tokasaurus)
```

### Phase 2: Actor Integration (Steps 5-6)
*"veRL's Actor does student forward passes with TrainableCache prepended"*

#### Step 5: Wrap the Actor model with `CacheAndModel`
**Files to modify**: `verl/verl/workers/fsdp_workers.py` or `verl/verl/workers/actor/dp_actor.py`

1. Load `TrainableCache` from checkpoint during Actor initialization
2. Freeze all base model parameters (`param.requires_grad = False`)
3. Wrap: `self.actor_module = CacheAndModel(cache, model)`
4. Optimizer targets `cache.parameters()` only
5. Use Cartridges' `FlexLlamaForCausalLM` model class (accepts `past_key_values=TrainableCache`)

**Config addition** to `ActorConfig`:
```yaml
actor_rollout_ref:
  actor:
    cartridge:
      enabled: true
      checkpoint_path: "path/to/cartridge.pt"
      model_cls: "cartridges.models.FlexLlamaForCausalLM"
```

#### Step 6: Teacher forward pass integration
**Options** (pick one):

**Option A: Use veRL's existing teacher service architecture**
The on-policy distillation recipe already has a ZMQ-based teacher service. The "teacher" would be the same base model served on a separate Tokasaurus/SGLang/vLLM instance with the full document in its prompt. This is the cleanest separation.

**Option B: In-process teacher forward pass**
Since the teacher is the same frozen model (just with full document context), run the teacher forward pass in the same Actor process. More GPU memory needed (full document KV cache) but simpler architecture.

For v1: **Option A** — reuse veRL's existing teacher infrastructure. The teacher is a separate inference server with the full document context. No cartridge.

### Phase 3: Cartridge Sync (Step 7)
*"After each optimizer step, push updated Cartridge to Tokasaurus"*

#### Step 7: Cartridge hot-reload mechanism

After `optimizer.step()`:
1. `cache.save("/tmp/verl_cartridge_latest.pt")`
2. Tokasaurus loads via `cartridges: [{id: "/tmp/verl_cartridge_latest.pt", source: "local", force_redownload: true}]`

Hook location: in the distillation trainer's main loop, after `update_policy()`.

### Phase 4: Tests & Config (Steps 8-9)

#### Step 8: Tests
**File**: `verl/tests/workers/rollout/test_tokasaurus_rollout.py`
- Unit: request formatting, response parsing (mock HTTP)
- Integration: rollout → student forward → teacher forward → KL loss → backward → cache params change

#### Step 9: Config & documentation
**File**: `verl/examples/cartridge_distill_training.yaml`

```yaml
actor_rollout_ref:
  model:
    path: meta-llama/Llama-3.2-3B-Instruct
    external_lib: cartridges
    model_cls: cartridges.models.FlexLlamaForCausalLM
  
  actor:
    cartridge:
      enabled: true
      checkpoint_path: "hazyresearch/cartridge-wauoq23f"
    optimizer: adam
    lr: 1e-3
  
  rollout:
    name: tokasaurus
    mode: async
    temperature: 0.7
    custom:
      tokasaurus_url: "http://localhost:10210"
      cartridges:
        - id: "/tmp/verl_cartridge_latest.pt"
          source: "local"
          force_redownload: true
    agent:
      num_workers: 4

  teacher:
    # Teacher = same model with full document context (no cartridge)
    server_ip: 127.0.0.1
    server_port: 15555
```

**README**: `verl/verl/workers/rollout/tokasaurus_rollout/README.md`

---

## Key Interfaces (from existing code)

### Tokasaurus endpoints
| Endpoint | Purpose | Request type |
|----------|---------|-------------|
| `POST /custom/cartridge/completions` | Student rollout with Cartridge | `CartridgeCompletionsRequest` (prompt as token IDs + cartridges array) |
| `GET /ping` | Health check | — |
| `GET /v1/models` | Model verification | — |

### Cartridges code we reuse directly
| Component | File | What it does |
|-----------|------|-------------|
| `TrainableCache` | `cartridges/cache.py` | `nn.Parameter` KV tensors, `.save()`, `.from_pretrained()`, `.clear()` |
| `CacheAndModel` | `cartridges/train.py` | Wraps frozen model + cache, passes `past_key_values=cache` |
| `FlexLlamaForCausalLM` | `cartridges/models/llama/` | Modified Llama that accepts `TrainableCache` as `past_key_values` |
| KL loss computation | `cartridges/train.py` lines 383-395 | `KL(teacher_logprobs, student_logits)` |

### veRL code we reuse/adapt
| Component | File | What it does |
|-----------|------|-------------|
| `RolloutReplica` | `verl/workers/rollout/replica.py` | Base class for rollout servers |
| `RolloutReplicaRegistry` | `verl/workers/rollout/replica.py` | Registration mechanism (`name: tokasaurus`) |
| `AsyncLLMServerManager` | `verl/experimental/agent_loop/agent_loop.py` | Load balancing, sticky sessions |
| `TokenOutput` | `verl/workers/rollout/replica.py` | Return type from `generate()` |
| `dp_actor.compute_log_prob()` | `verl/workers/actor/dp_actor.py` | Student forward pass (adapt for TrainableCache) |
| `OnPolicyDistillTrainer` | `recipe/gkd/` (external) | Distillation training loop (adapt for Cartridges) |

---

## Files to Create

| File | Purpose |
|------|---------|
| `verl/verl/workers/rollout/tokasaurus_rollout/__init__.py` | Package init |
| `verl/verl/workers/rollout/tokasaurus_rollout/async_tokasaurus_server.py` | `TokasaurusHttpServer` + `TokasaurusReplica` |
| `verl/tests/workers/rollout/test_tokasaurus_rollout.py` | Unit + integration tests |
| `verl/verl/workers/rollout/tokasaurus_rollout/README.md` | Documentation |
| `verl/examples/cartridge_distill_training.yaml` | Example config |

## Files to Modify

| File | Change |
|------|--------|
| `verl/verl/workers/rollout/replica.py` | Register `"tokasaurus"` in `RolloutReplicaRegistry` |
| `verl/verl/workers/actor/dp_actor.py` | Add Cartridge support to `compute_log_prob()` |
| `verl/verl/workers/fsdp_workers.py` | Load `TrainableCache`, wrap in `CacheAndModel`, configure optimizer |

---

## Execution Order

| # | Phase | What | Depends On | Difficulty |
|---|-------|------|------------|------------|
| 1 | Rollout Bridge | `TokasaurusHttpServer` + `TokasaurusReplica` + registry | Nothing | Medium |
| 2 | Rollout Bridge | Unit tests for Phase 1 | Phase 1 | Easy |
| 3 | Actor Integration | Load `TrainableCache`, wrap model, modify optimizer | Nothing (parallel) | Hard |
| 4 | Actor Integration | `compute_log_prob` with Cartridge prepended | Phase 3 | Medium |
| 5 | Cartridge Sync | Save/reload after optimizer step | Phase 1 + 3 | Easy |
| 6 | End-to-end | Integration test: full distillation loop | All above | Medium |
| 7 | Config/Docs | YAML config, README | All above | Easy |

---

## Verification & Demo Plan

### Models & Pre-trained Assets

| Asset | Source |
|-------|--------|
| **Llama-3.2-3B-Instruct** | HuggingFace (~6.4 GB VRAM) |
| **Pre-trained Cartridge** | `hazyresearch/cartridge-wauoq23f` (HF/WandB, LongHealth, 2048 token KV) |
| **Synthetic dataset** | `hazyresearch/arxiv_synthesize_qwen-qwen3-4b_n8192-0` (HF) |

### Demo Tiers

| Tier | What | Proves |
|------|------|--------|
| **A** | Unit tests pass (mock HTTP) | Request/response format correct |
| **B** | Rollout through Tokasaurus → valid TokenOutput | Bridge works, Cartridge injection works |
| **C** | Student forward with TrainableCache → differentiable log_probs | Actor integration works |
| **D** | Full loop: rollout → student logits → teacher logits → KL loss → backward → cache updates → sync → rollout again | On-policy Cartridge distillation works end-to-end |

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **FlexLlama vs standard HF model** | FlexLlama inherits `PreTrainedModel` — test compatibility early |
| **FSDP + TrainableCache** | Cache is tiny, skip FSDP for it. Only model (frozen) gets FSDP. |
| **Teacher needs full document in context** | May exceed single-GPU memory for very long docs. Use separate teacher server. |
| **geoff/cartridges unmerged** | Pin commit. Health check in `launch_servers()`. |
| **Cache state between samples** | Call `cache.clear()` between micro-batches |

## No Cartridge Staleness
Base model stays frozen in both Tokasaurus and veRL. Only Cartridge KV tensor updates. No staleness.
