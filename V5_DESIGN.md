# deepspec-v5:OPD 最优训练流水线

[English](./V5_DESIGN_en.md) | [简体中文](./V5_DESIGN.md)

2026-08-12 从 train-v4/deepspec-opd 派生。**每个默认值都有实测依据**(见下),
测量环境:node-h200-1,target=DeepSeek-V4-Flash-0731(SGLang TP4,GPU 0-3),
训练 GPU 4-7,数据集 5632 条医疗对话(序列 p50≈15.6k)。

入口配置:`config/dspark/dspark_dskv4flash_v5.py`

```bash
python train.py --config config/dspark/dspark_dskv4flash_v5.py
```

## 默认值与实测依据(gbs=64 单步,0731 权重)

| 默认 | 取值 | 依据 |
|---|---|---|
| backend | mcore + megatron trainer | HF 路线无 EP,gbs=64 即 129G+ 贴顶;FSDP2 路线(v3)另测 |
| **EP** | **4(必需)** | EP=1 实测 OOM(135.5G 仍不够);EP=4 训练侧省 43.8G 且纯计算快 35% |
| grouped GEMM | 开 | 关闭无显存收益;bench_ep 确认 TEGroupedMLP 生效 |
| **recompute_moe_layer** | **False** | 实测计算侧时间 -48%,峰值显存反而 -7GB(反直觉,双赢) |
| **fp8** | **"hybrid"**(blockwise recipe) | 实测 -8% 时间、-3GB 显存、loss 差 1e-4;fp8_param=False 保守起步 |
| **rollout 缓存** | **默认开** | 0731 下 27.3→5.3 min = **5.2x**;loss 相对 n=5 基线 -0.6σ 不显著(旧权重口径) |
| local_batch_size | 1(定死) | lbs=2 实测 OOM(143G) |
| torch_compile | False(结构性) | 与 EP 的 all_to_all 互斥:MoE 分发 token 数动态,compile 固化 split 尺寸即炸 |
| opd_max_new_tokens | 512 | 信号设计项:128=半价多截断;1000=+10% 价近乎不截断(旧权重口径,0731 分布已变:p50=404、24% 打满 512) |

## 架构(2026-08-13 重构,两天基准实验的直接产物)

### 两阶段训练(核心)

```
阶段一  scripts/prebuild_rollout_cache.py     阶段二  train.py
┌─────────────────────────────┐              ┌─────────────────────────────┐
│ 纯 rollout 建 gen_ids 缓存     │   缓存目录    │ 全 warm 训练                  │
│ 无 NCCL、不占训练 GPU          │ ──────────→ │ 数据侧只剩特征导出              │
│ 服务死锁 = 变慢,可断点续       │              │ gbs=512 一步 ≈ 40min          │
└─────────────────────────────┘              └─────────────────────────────┘
```

动机:训练的 EP all_to_all 是同步集合通信,任何 rank 的数据长尾都让全体等待;
而特征服务会间歇死锁(两天内实测 6 次)。耦合时服务打嗝 > NCCL 窗口 = 训练崩
(实测三种崩法)。解耦后,阶段一对故障完全免疫(无限重试+续跑),阶段二 warm
下最长等待 = 单次导出,远离一切超时窗口。

### 统一特征客户端 `deepspec/data/feature_client.py`

rollout / 缓存重放 / teacher-forced 三条请求路径此前重试逻辑分散、三次事故
分别打补丁;现收敛为 `FeatureClient`(一处重试策略、一处 bf16 编解码、一处
端点轮询),loader 与 trainer 都走它,消除了 loader→trainer 反向依赖。
旧函数(`fetch_target_features`/`rollout_target_features`/`_decode_hidden`)
签名保留,内部委托 client。

### 运维件 `scripts/ops/`

- `svc_sentinel.sh`:服务自愈哨兵。忙/死鉴别(探测超时时看日志里真实 batch 行,
  死锁服务只刷 state-deleted 余波);SIGTERM→8s→kill -9(死锁服务连优雅退出都挂)。
- `gate_train_behind_service.sh`:训练只在服务连续 3×200 后放行
  (防 JIT 期被打崩、防驱动抢跑)。

### 测试 `tests/test_rollout_loader_paths.py`

不联网对拍三条取数路径:相同输入下产出逐位一致、loss 区口径正确、
prefill 裁剪无越界(哨兵值检测)。

## 继承自 v4 的健壮性修复(2026-08-12)

1. 缓存命中路径失败自动回退 rollout(防单次 ConnectionError 打崩训练)
2. 缓存 key 含 `max_length`(budget 语义)与 `weight_version=target 路径`(换权重自动失效)
3. 等待计时无盲区 + 缓存命中率入日志

## 已知约束与坑

- **服务重启后首请求触发 DeepGEMM JIT 预编译(10-20 min)**,不是故障
- target 特征服务**不可复现约 15%**(重尾,随序列位置累积,loss 区在最不稳区域)——
  训练信号质量问题,与本流水线无关但要知道
- 特征导出是每样本约 6.4s 的地板(634MiB @ 99MiB/s);缓存把 decode 消掉后**导出就是
  下一个瓶颈**,要再快需改 SGLang 返回路径
- 并发调不动:服务 C=8 即饱和,`opd_prefetch` 不是杠杆
- 乱序消费 × 断点恢复:恢复偏移最多错位 window×ranks 个样本(SGD 无害)
- micro-batch 等权:loss token 数样本间差 75 倍,短回复样本每 token 权重更高(设计选择)

## 待回填(实验进行中)

- [ ] gbs=512 × 2 步:组合(fp8+norec)cold / warm vs 基线 warm A/B
- [ ] 2-epoch 完整训练(epoch2 缓存复利 + loss 曲线)
- [ ] 20 步 loss 曲线:缓存 vs 无缓存(回答"偏移是否累积")
- [ ] v3 FSDP2 与 v4 HF 路线对照数字

## shm 直通导出(2026-08-27,已上生产)

特征导出 6.4s/样本地板已破:大块 hidden 写 /dev/shm(容器 IpcMode=host 共享),
JSON 只传 "shm:tag:rows:cols:path" 指针。单请求 9.3→1.44s(6.5x),
gbs=64 warm 步时 5.3→2.1min(2.5x),数值无损(double 余弦 0.9999)。
服务端改动在 sglang-dspark-src 副本(PYTHONPATH 加载),开关 SGLANG_DSPARK_EXPORT_SHM。
三个正确性关键:确定性命名(TP 各 rank 重复写)、tmp+原子 replace(半截文件竞态)、
客户端读后删 + keeper 30min 兜底(泄漏)。
注意:端口勿用 30000-32767(k8s NodePort 范围,kube-proxy 会占;30000 是侥幸)。
