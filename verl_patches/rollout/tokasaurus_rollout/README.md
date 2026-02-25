# Tokasaurus Rollout for veRL

Connects veRL's rollout infrastructure to an external [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) inference server with **Cartridge KV tensor injection** support.

## Why

Standard inference engines (vLLM, SGLang) accept **text** and compute a KV cache from scratch via prefill. A [Cartridge](https://github.com/HazyResearch/cartridges) is already a pre-computed KV cache tensor. Tokasaurus (geoff/cartridges branch) has a custom endpoint that **bypasses prefill** and injects the Cartridge's KV tensors directly into PagedAttention memory.

This module lets veRL generate rollouts through Tokasaurus so the model can use the Cartridge during generation — enabling on-policy Cartridge distillation.

## Architecture

```
AsyncLLMServerManager.generate()
        │
        ▼
TokasaurusHttpServer (Ray actor, no GPU)
        │  HTTP POST
        ▼
Tokasaurus /custom/cartridge/completions
        │  bypass prefill, inject Cartridge KV
        ▼
    TokenOutput(token_ids, log_probs)
```

## Configuration

Set `name: tokasaurus` in your rollout config and provide Tokasaurus connection details via `custom`:

```yaml
actor_rollout_ref:
  rollout:
    name: tokasaurus
    mode: async
    temperature: 0.7
    custom:
      tokasaurus_url: "http://localhost:10210"
      cartridges:
        - id: "/tmp/cartridge.pt"
          source: "local"
          force_redownload: true
      max_retries: 5
      base_timeout: 120
```

### Custom fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tokasaurus_url` | str | `http://localhost:10210` | Tokasaurus server URL |
| `cartridges` | list[dict] | `[]` | Cartridge specs: `{id, source, force_redownload}` |
| `max_retries` | int | `5` | Max HTTP retries with exponential backoff |
| `base_timeout` | float | `120.0` | Base HTTP timeout in seconds |

### Cartridge sources

| Source | `id` format | Example |
|--------|------------|---------|
| `local` | File path | `/tmp/cartridge.pt` |
| `wandb` | WandB run ID | `hazyresearch/cartridge-wauoq23f` |
| `huggingface` | HF repo ID | `hazyresearch/cartridge-wauoq23f` |
| `s3` | S3 URI | `s3://bucket/cartridge.pt` |

## Prerequisites

1. **Tokasaurus** running on the `geoff/cartridges` branch:
   ```bash
   git clone https://github.com/ScalingIntelligence/tokasaurus
   cd tokasaurus && git checkout geoff/cartridges
   pip install -e .
   toka model=meta-llama/Llama-3.2-3B-Instruct kv_cache_num_tokens='(256 * 1024)'
   ```

2. **veRL** with this module available (it's registered automatically via `RolloutReplicaRegistry`).

## Testing

```bash
# Unit tests (no GPU, no Tokasaurus needed)
pytest tests/workers/rollout/test_tokasaurus_rollout.py -v
```

## Key differences from SGLang/vLLM rollout

| | SGLang/vLLM | Tokasaurus |
|---|---|---|
| Server lifecycle | Launched in-process by veRL | Runs externally, veRL connects via HTTP |
| GPU usage by actor | Shares GPU with training | No GPU (HTTP proxy only) |
| Cartridge support | ❌ | ✅ via `cartridges` array in request |
| Weight sync | NCCL broadcast from trainer | N/A (base model frozen, only Cartridge changes) |
