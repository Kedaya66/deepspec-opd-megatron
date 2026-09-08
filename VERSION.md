# DeepSpec OPD Megatron v5

[English](./VERSION.md) | [简体中文](./VERSION_zh.md)

## Positioning

v5 was derived from train-v4/deepspec-opd on August 12, 2026. It targets stable
single-node H200 training of a DSpark draft model for DeepSeek-V4-Flash. The
measured setup uses four training GPUs, four GPUs for the target feature
service, and 5,632 long-form medical conversations.

## Default Training Configuration

| Item | Default | Rationale |
| --- | --- | --- |
| Backend | Megatron-Core + Megatron trainer | Supports MoE Expert Parallelism and lowers draft-model memory pressure |
| EP / TP | EP=4 / TP=1 | Splits 256 experts across training GPUs; the three-layer draft does not need TP |
| Grouped GEMM | Enabled | Uses Transformer Engine to avoid inefficient per-expert small GEMMs |
| Precision | BF16 + FP8 hybrid/blockwise | Reduced runtime and memory in measurements while retaining conservative parameter precision |
| MoE recompute | Disabled | Reduced compute time by about 48% and peak memory by about 7 GB |
| Local / global batch | 1 / 512 | Local batch size 2 caused OOM on the target hardware |
| Rollout cache | Enabled | Reduced an older-weight baseline step from 27.3 to 5.3 minutes, about 5.2x |
| OPD max new tokens | 512 | Balances throughput, truncation rate, and training signal |

## Key Improvements

### Two-stage training

scripts/prebuild_rollout_cache.py creates a recoverable cache without NCCL
training communication, then train.py performs warm training. Target-service
tail latency or restarts no longer directly stall synchronous EP collectives,
and the training stage avoids serial decode cost.

### Unified feature client

deepspec/data/feature_client.py centralizes retry policy, BF16 encoding and
decoding, and endpoint rotation for rollout, cache replay, and teacher-forced
paths. Compatibility entry points delegate to the same client.

### Shared-memory feature transport

Large hidden states can be transported through /dev/shm while JSON carries only
a reference. Production measurements reduced one request from 9.3 to 1.44
seconds and a warm step from 5.3 to 2.1 minutes. This path requires host IPC
sharing and server-side SGLANG_DSPARK_EXPORT_SHM support.

### Service recovery and startup gating

- scripts/ops/svc_sentinel.sh distinguishes busy services from deadlocks and performs staged restarts
- scripts/ops/gate_train_behind_service.sh starts training only after consecutive healthy probes
- Rollout-cache keys include the generation budget and target weight version

## Known Limitations

- Default configs contain environment-specific model, dataset, cache, and checkpoint paths
- The first request after a service restart may spend 10 to 20 minutes compiling DeepGEMM kernels
- Target features have heavy-tail latency and limited reproducibility that can affect the training signal
- Dynamic MoE all-to-all token splits are incompatible with torch.compile in this setup
- Shared-memory transport requires client cleanup and a keeper for crash leftovers
- Micro-batches are equally weighted rather than weighted by loss-token count

## Status

v5 includes OPD, Megatron-Core EP, FP8, rollout caching, shared-memory feature
export, and service recovery. Full multi-epoch convergence comparisons,
long-term cache-shift analysis, and FSDP2/HF baseline comparisons remain open.

See [V5_DESIGN_en.md](./V5_DESIGN_en.md) for detailed evidence and history.
