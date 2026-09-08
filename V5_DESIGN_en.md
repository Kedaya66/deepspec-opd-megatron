# deepspec-v5: Optimized OPD Training Pipeline

[English](./V5_DESIGN_en.md) | [简体中文](./V5_DESIGN.md)

v5 was derived from train-v4/deepspec-opd on August 12, 2026. Every default
below is backed by measurements on node-h200-1 with DeepSeek-V4-Flash-0731 as
the target, SGLang TP4 on GPUs 0-3, training on GPUs 4-7, and 5,632 medical
conversations with a sequence-length p50 of about 15.6k.

Primary configuration: config/dspark/dspark_dskv4flash_v5.py

```bash
python train.py --config config/dspark/dspark_dskv4flash_v5.py
```

## Measured Defaults

The single-step measurements in this section use global batch size 64 and the
0731 target weights.

| Setting | Value | Evidence |
| --- | --- | --- |
| Backend | mcore + Megatron trainer | The HF path has no EP and approaches 129 GB at gbs=64; FSDP2 was measured separately |
| Expert Parallelism | EP=4, required | EP=1 OOMs above 135.5 GB; EP=4 saves 43.8 GB and makes pure compute 35% faster |
| Grouped GEMM | Enabled | Disabling it provides no memory benefit; bench_ep confirms TEGroupedMLP is active |
| recompute_moe_layer | False | Compute time falls 48% and peak memory unexpectedly falls by 7 GB |
| FP8 | hybrid, blockwise recipe | Runtime falls 8%, memory falls 3 GB, and loss differs by about 1e-4; fp8_param remains false |
| Rollout cache | Enabled | Step time falls from 27.3 to 5.3 minutes, about 5.2x; loss shift versus the n=5 baseline is not significant |
| local_batch_size | 1 | A local batch size of 2 OOMs at 143 GB |
| torch_compile | False | Dynamic MoE all-to-all split sizes conflict with compile-time specialization |
| opd_max_new_tokens | 512 | A signal-design compromise; 128 truncates often while 1000 costs about 10% more with little truncation |

## Architecture

### Two-stage training

```text
Stage 1: prebuild_rollout_cache.py       Stage 2: train.py
+--------------------------------+      +-------------------------------+
| Build a gen_ids rollout cache  | ---> | Fully warm training           |
| No NCCL or training GPUs       |      | Only feature export remains   |
| Retry and resume on failures   |      | gbs=512 step about 40 minutes |
+--------------------------------+      +-------------------------------+
```

EP all-to-all is synchronous, so tail latency on any rank stalls every rank.
The target feature service deadlocked six times during two days of testing.
When rollout and training were coupled, a service pause longer than the NCCL
window crashed training in three observed ways.

The split pipeline isolates these failures. Stage 1 retries indefinitely and
resumes without involving NCCL. During a warm Stage 2 run, the longest data
wait is one feature export, keeping it far from collective timeouts.

### Unified FeatureClient

Rollout, cache replay, and teacher-forced requests previously had independent
retry logic and accumulated separate incident fixes. They now use
deepspec/data/feature_client.py for one retry policy, one BF16 codec, and one
endpoint-rotation implementation. Both loaders and trainers depend on the
client, removing the reverse loader-to-trainer dependency.

The old fetch_target_features, rollout_target_features, and _decode_hidden
interfaces remain compatible and delegate to FeatureClient.

### Operational tooling

- scripts/ops/svc_sentinel.sh distinguishes a busy service from a deadlock by
  checking real batch activity after probe timeouts. It escalates from SIGTERM
  to SIGKILL because a deadlocked service may not exit cleanly.
- scripts/ops/gate_train_behind_service.sh releases training only after three
  consecutive HTTP 200 probes, protecting the service during JIT warmup and
  preventing driver races.

### Offline consistency test

tests/test_rollout_loader_paths.py compares rollout, cache replay, and
teacher-forced data paths without network access. It verifies identical output
for identical input, correct loss regions, and bounded prefill trimming with
sentinel values.

## Reliability Fixes Inherited from v4

1. A failed cache-hit request automatically falls back to rollout.
2. Cache keys include max_length as the generation budget and the target weight
   path as weight_version, so either change invalidates stale entries.
3. Wait-time accounting has no blind spots, and cache hit rate is logged.

## Constraints and Pitfalls

- The first request after a service restart triggers 10-20 minutes of DeepGEMM JIT compilation; this is not a failure.
- Target feature output is roughly 15% non-reproducible, with heavy-tail error accumulating by sequence position in the least stable loss region.
- Before shared memory, exporting 634 MiB at about 99 MiB/s imposed a 6.4-second per-sample floor.
- Service concurrency saturates at about C=8; increasing opd_prefetch is not a throughput lever.
- Out-of-order consumption plus resume can shift at most window times ranks samples, which is acceptable for SGD.
- Micro-batches are equally weighted even though loss-token counts vary by up to 75x.

## Open Experiments

- [ ] Two gbs=512 steps comparing FP8 plus no-recompute cold/warm runs against the warm baseline
- [ ] Full two-epoch training to measure second-epoch cache compounding and loss curves
- [ ] Twenty-step cache-versus-no-cache loss curves to detect accumulated shift
- [ ] FSDP2 v3 comparison against the v4 HF path

## Shared-memory Direct Export

This path entered production on August 27, 2026 and removed the 6.4-second
feature-export floor. Large hidden tensors are written to /dev/shm with host IPC
sharing, while JSON carries a shm reference. One request improved from 9.3 to
1.44 seconds, and a warm gbs=64 step improved from 5.3 to 2.1 minutes, with
double-precision cosine similarity of 0.9999.

The server changes live in the sglang-dspark-src copy loaded through PYTHONPATH
and are controlled by SGLANG_DSPARK_EXPORT_SHM. Correctness depends on:

1. deterministic names because TP ranks may write the same result,
2. temporary files followed by atomic replacement to avoid partial-read races,
3. client deletion after reading plus a 30-minute keeper fallback for leaks.

Avoid ports 30000-32767 because they overlap the Kubernetes NodePort range and
may be occupied by kube-proxy.
