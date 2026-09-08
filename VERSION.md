# DeepSpec OPD Megatron v5

## 版本定位

v5 于 2026-08-12 从 `train-v4/deepspec-opd` 派生，目标是在单节点 H200 环境中
稳定训练 DeepSeek-V4-Flash 的 DSpark draft model。训练侧使用 4 张 GPU，target
feature 服务使用另外 4 张 GPU；默认数据基线为 5,632 条长序列医疗对话。

## 默认训练方案

| 项目 | 默认值 | 原因 |
| --- | --- | --- |
| Backend | Megatron-Core + Megatron trainer | 支持 MoE Expert Parallel，降低大 draft model 显存压力 |
| EP / TP | EP=4 / TP=1 | 256 个专家按训练 GPU 切分；三层 draft 不依赖 TP 扩展 |
| Grouped GEMM | 开启 | 通过 Transformer Engine 避免专家小 GEMM 退化 |
| Precision | BF16 + FP8 hybrid/blockwise | 实测减少训练时间和显存，参数仍保持保守精度 |
| MoE recompute | 关闭 | 实测计算时间约降 48%，峰值显存同时下降约 7 GB |
| Local / global batch | 1 / 512 | local batch 2 会在目标硬件上 OOM |
| Rollout cache | 开启 | 旧权重基线下单步由 27.3 分钟降至 5.3 分钟，约 5.2 倍 |
| OPD max new tokens | 512 | 在吞吐、截断率和训练信号之间折中 |

## v5 关键改进

### 两阶段训练

`scripts/prebuild_rollout_cache.py` 先在无 NCCL 训练通信的阶段生成可恢复缓存，
`train.py` 再进行 warm training。这样 target 服务的长尾或重启不会直接拖垮同步
EP collective，训练阶段也不再承担串行 decode 成本。

### 统一特征客户端

`deepspec/data/feature_client.py` 统一 rollout、缓存重放和 teacher-forced 三条路径的
重试、BF16 编解码与端点轮询。旧入口保持兼容并委托给该客户端。

### 共享内存特征传输

大块 hidden states 可通过 `/dev/shm` 传输，JSON 仅携带引用。生产实测单请求由
9.3 秒降至 1.44 秒，warm step 由 5.3 分钟降至 2.1 分钟。该路径要求容器共享
host IPC，并启用服务端 `SGLANG_DSPARK_EXPORT_SHM` 支持。

### 服务自愈与启动门禁

- `scripts/ops/svc_sentinel.sh` 区分繁忙和死锁服务，并执行分级重启
- `scripts/ops/gate_train_behind_service.sh` 要求服务连续健康后才放行训练
- rollout cache key 纳入生成预算和目标权重版本，配置变化会自动失效

## 已知限制

- 默认配置含环境相关的绝对模型、数据、缓存和 checkpoint 路径
- target 服务重启后的首个请求可能触发 10 到 20 分钟 DeepGEMM JIT
- target 特征仍存在重尾和一定不可复现性，可能影响训练信号
- MoE all-to-all 的动态 token split 与 `torch.compile` 不兼容，默认关闭 compile
- 共享内存传输要求客户端及时清理，并由 keeper 处理异常退出后的泄漏
- 当前方案按 micro-batch 等权，而非按 loss token 数加权

## 当前状态

v5 已具备 OPD、Megatron-Core EP、FP8、rollout cache、共享内存特征导出和服务自愈
能力。完整多 epoch 收敛对比、缓存偏移长期影响及 FSDP2/HF 路线对照仍需继续验证。

更完整的实验依据和历史记录见 [V5_DESIGN.md](./V5_DESIGN.md)。
