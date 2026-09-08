# DeepSpec OPD Megatron

[English](./README.md) | [简体中文](./README_zh.md)

**基于 Megatron-Core、专家并行和 FP8，高效稳定训练大规模推测解码 Draft Model 的 OPD 流水线。**

DeepSpec OPD Megatron 是基于 DeepSpec 的生产训练分支，面向
DeepSeek-V4-Flash 的在线策略蒸馏（OPD）和推测解码 draft model 训练。该版本将
Megatron-Core MoE、Expert Parallel、Transformer Engine FP8、rollout 缓存与共享
内存特征传输组合成适合多张 H200 GPU 的训练流水线。

当前仓库版本为 **v5**。生产配置、实测默认值、性能数据和限制见
[VERSION_zh.md](./VERSION_zh.md)，设计过程与实验记录见 [V5_DESIGN.md](./V5_DESIGN.md)。

> 本分支包含针对内部 DeepSeek-V4-Flash 服务和数据路径的配置。公开或迁移部署时，
> 请先替换模型、数据、缓存、checkpoint 和 SGLang 服务地址；密钥只放在本地环境中。

## v5 快速开始

```bash
python -m pip install -r requirements.txt
bash run_svc_opd.sh
python scripts/prebuild_rollout_cache.py \
  --config config/dspark/dspark_dskv4flash_v5.py
bash run_train_opd.sh
```

核心入口配置是 config/dspark/dspark_dskv4flash_v5.py。默认训练拓扑为单节点
4 张训练 GPU、EP=4、TP=1；target 服务使用另 4 张 GPU。运行前必须根据机器环境
检查配置中的绝对路径和 SGLang 服务地址。

## v5 关键改进

- 两阶段训练：先生成可恢复 rollout 缓存，再进行 warm training
- 统一特征客户端：集中处理重试、BF16 编解码和端点轮询
- Megatron-Core MoE：使用 EP=4、Grouped GEMM 和精度感知优化器
- FP8 hybrid/blockwise：在目标硬件上降低训练时间和显存
- 共享内存特征传输：绕过大块 hidden states 的 JSON 序列化瓶颈
- 服务自愈与门禁：死锁检测、分级重启及连续健康检查

## 仓库组成

| 路径 | 内容 |
| --- | --- |
| deepspec/data | 在线特征客户端、预取、rollout 与 target cache 数据管线 |
| deepspec/modeling | DSpark、DFlash、Eagle3 及 DeepSeek-V4 Megatron 实现 |
| deepspec/trainer/megatron | 并行组、DDP、优化器、checkpoint 和训练器 |
| config/dspark | 通用和 DeepSeek-V4-Flash 专用训练配置 |
| scripts/ops | 服务哨兵与训练启动门禁 |
| scripts/data | 数据生成、target cache 和 SGLang 特征服务工具 |
| tests | FSDP2、Megatron parity/TP 和 rollout 路径测试 |
| eval.py / eval_datasets | 推测解码评测入口与基准数据 |

## 验证

```bash
python -m pytest tests/test_rollout_loader_paths.py
python -m pytest tests/test_mcore_parity.py tests/test_mcore_tp.py
```

分布式 smoke test 和完整训练依赖 CUDA、Megatron-Core、Transformer Engine、
对应模型权重及可用的 target feature 服务。

## 支持的算法

上游 DeepSpec 支持 [DSpark](https://arxiv.org/abs/2607.05147)、
[DFlash](https://arxiv.org/abs/2602.06036) 和
[Eagle3](https://arxiv.org/abs/2503.01840)。通用数据准备、训练、评测流程以及已发布
checkpoint 列表见 [英文 README](./README.md#upstream-deepspec)。

## 许可证与引用

项目采用 [MIT License](./LICENSE)。第三方代码归属见 [NOTICE](./NOTICE)。
使用 DSpark 研究成果时，请采用英文 README 中的[论文引用](./README.md#citation)。
