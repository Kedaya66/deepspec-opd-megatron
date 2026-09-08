import os

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

from deepspec.trainer.base_trainer import _clip_grad_norm_fsdp2, _fsdp2_no_sync
from deepspec.trainer.ckpt_manager import _validate_optimizer_state_finite
from deepspec.utils.optim import BF16Optimizer


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs):
        return torch.nn.functional.silu(self.linear(inputs))


class TinyModel(torch.nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(width), Block(width)])
        self.head = torch.nn.Linear(width, 1, bias=False)

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return self.head(inputs)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(1234)

    device = torch.device("cuda", local_rank)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    kwargs = dict(mesh=mesh, offload_policy=CPUOffloadPolicy())
    for block in model.blocks:
        fully_shard(block, **kwargs)
    fully_shard(model, reshard_after_forward=False, **kwargs)

    optimizer = BF16Optimizer(
        model, lr=1e-3, total_steps=2, warmup_ratio=0.0
    )
    for micro_step in range(2):
        inputs = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
        context = _fsdp2_no_sync(model) if micro_step == 0 else torch.no_grad()
        if micro_step == 0:
            with context:
                model(inputs).float().square().mean().backward()
        else:
            model(inputs).float().square().mean().backward()

    grad_norm = _clip_grad_norm_fsdp2(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm)
    optimizer.step()
    optimizer_state = optimizer.state_dict()
    optimizer.load_state_dict(optimizer_state)
    _validate_optimizer_state_finite(optimizer)

    state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if dist.get_rank() == 0:
        assert state and all(torch.isfinite(value).all() for value in state.values())
        checksum_value = sum(value.float().sum().item() for value in state.values())
    else:
        assert not state
        checksum_value = 0.0
    checksum = torch.tensor(checksum_value, device=device)
    dist.broadcast(checksum, src=0)
    assert torch.isfinite(checksum)
    if dist.get_rank() == 0:
        print(f"FSDP2 smoke OK: grad_norm={grad_norm.item():.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
