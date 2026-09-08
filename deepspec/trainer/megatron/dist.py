"""Megatron-Core 并行状态初始化与数据并行维度工具。

沿用项目 init_dist 的进程组建立方式(train.py 每卡 spawn 一个 worker、
环境变量 RANK/WORLD_SIZE 是节点级语义),在其上再建立 megatron 的
TP/PP/CP/EP/DP 并行组。
"""

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from deepspec.utils import init_dist


def init_megatron_parallel_state(
    # 2026-08-13 timeout 60->240:EP 子组继承全局 PG 的 timeout(megatron 的 distributed_timeout_minutes 在本版 mcore 未传导到子组,实测 watchdog 报 3600000ms)。OPD 数据管线等 rollout 属合法长等待。
    local_rank, megatron_cfg, timeout_minutes: int = 240, model_cfg=None, seed: int = 42
):
    """初始化 torch.distributed 与 megatron 并行组,返回 (device, global_rank, world_size)。

    并行度只有一个真实来源。model.backend=="mcore" 时模型自带 model.mcore
    那套并行选项,而模型构造里的 init_dspark_parallel 遇到"已初始化"会直接
    return —— 也就是说先建组的一方说了算。为免 EP 配了却静默失效,这里强制
    两份配置必须一致,不一致就报错而不是挑一个。
    """
    device, global_rank, world_size = init_dist(local_rank, timeout_minutes)
    tp = int(megatron_cfg.tensor_model_parallel_size)
    pp = int(megatron_cfg.pipeline_model_parallel_size)
    cp = int(megatron_cfg.context_parallel_size)
    ep = int(megatron_cfg.expert_model_parallel_size)

    is_mcore = False
    if model_cfg is not None:
        is_mcore = str(
            model_cfg.backend if "backend" in model_cfg else ""
        ).lower() == "mcore"

    if is_mcore:
        mc = dict(model_cfg.mcore) if "mcore" in model_cfg else {}
        m_tp = int(mc.get("tensor_model_parallel_size", 1))
        m_ep = int(mc.get("expert_model_parallel_size", 1))
        assert (m_tp, m_ep) == (tp, ep), (
            "train.megatron 与 model.mcore 的并行度不一致:"
            f"megatron(tp={tp}, ep={ep}) vs mcore(tp={m_tp}, ep={m_ep})。"
            "两边必须相同 —— 模型构造时 init_dspark_parallel 对已建组的情况直接"
            "返回,不一致会导致其中一份被静默忽略。"
        )
        assert (pp, cp) == (1, 1), (
            f"mcore draft 只有 3 层,PP/CP 无意义(收到 pp={pp}, cp={cp})"
        )
    elif (tp, pp, cp, ep) != (1, 1, 1, 1):
        raise NotImplementedError(
            "draft 模型是普通 HF 风格实现,尚未做 megatron 模型并行改造,"
            f"当前仅支持 TP=PP=CP=EP=1(收到 tp={tp}, pp={pp}, cp={cp}, ep={ep});"
            "要用 TP/EP 请设 model.backend='mcore'"
        )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        distributed_timeout_minutes=timeout_minutes,
    )
    # 注册 mcore 的各路 RNG 状态(data / tensor-model / expert-parallel)。
    # HF 实现不需要,但 mcore 的 ColumnParallelLinear(is_expert=True) 初始化时
    # 要 fork 'expert-parallel-rng',没注册就报
    # "Exception: cuda rng state expert-parallel-rng is not added"(实测)。
    # 它同时保证 TP 各 rank 拿到不同 RNG —— 被切开的权重不能是彼此的复制品。
    model_parallel_cuda_manual_seed(int(seed))
    return device, global_rank, world_size


def get_data_parallel_rank():
    return parallel_state.get_data_parallel_rank()


def get_data_parallel_world_size():
    return parallel_state.get_data_parallel_world_size()


def destroy_megatron_parallel_state():
    parallel_state.destroy_model_parallel()
