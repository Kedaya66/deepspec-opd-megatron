"""不联网:验证 RolloutPrefetchLoader 三条取数路径的组装一致性。

关键口径:packed 行数 = len(seq)-1(末 token 无 hidden),
缓存命中路径与 rollout 路径给定相同 gen_ids/特征时产出必须逐位一致。
直接跑:python tests/test_rollout_loader_paths.py
"""
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepspec.data.opd_rollout_loader import RolloutPrefetchLoader  # noqa: E402

N_LAYERS, H, PLEN, GEN = 3, 16, 40, 7
COLS = (N_LAYERS + 1) * H

torch.manual_seed(0)
PACKED = torch.randn(PLEN + GEN - 1, COLS).to(torch.bfloat16)  # rollout 口径行数
GEN_IDS = list(range(9000, 9000 + GEN))
ids = torch.arange(1, PLEN + GEN + 1, dtype=torch.long)
lm = torch.zeros(PLEN + GEN, dtype=torch.long)
lm[PLEN:] = 1


class FakeClient:
    """桩掉 FeatureClient:rollout 返回固定 gen+packed;prefill 多给 1 行哨兵,
    验证调用方 rows= 裁剪正确(哨兵混入 = 裁剪 bug)。"""

    def rollout(self, prompt, *, max_new_tokens, temperature, top_p):
        return list(GEN_IDS), PACKED.clone()

    def prefill_features(self, seq, *, rows=None):
        full = torch.cat([PACKED, torch.full((1, COLS), 999.0).to(torch.bfloat16)], 0)
        return full[:rows] if rows is not None else full


def run(cache_dir, preseed):
    shutil.rmtree(cache_dir, ignore_errors=True)
    loader = RolloutPrefetchLoader(
        [], endpoints=["http://x"], n_target_layers=N_LAYERS,
        max_length=10000, max_new_tokens=512,
        cache_dir=cache_dir, weight_version="/w")
    loader.client = FakeClient()
    if preseed:
        loader.cache.put(ids[:PLEN].tolist(), GEN_IDS)
    return loader._rollout_one(ids.clone(), lm.clone())


miss = run("/tmp/v5_paths_miss", preseed=False)   # 路径 2(rollout)
hit = run("/tmp/v5_paths_hit", preseed=True)      # 路径 1(缓存重放)

ok = True
for n, a, b in zip(("seq", "loss_mask", "th", "tl"), miss, hit):
    same = torch.equal(a, b)
    ok &= same
    print(f"  {n:<10} {tuple(a.shape)} 逐位相同={same}")

seq, lmask, th, tl = miss
assert int(torch.nonzero(lmask)[0]) == PLEN, "loss 区起点错"
assert int(lmask.sum()) == GEN - 1, "loss token 数错(末 token 应排除)"
assert bool(th[-1].abs().sum() == 0), "th 末行应为零(无 hidden)"
assert bool((th.float() != 999).all()), "哨兵 999 混入 => prefill rows 裁剪 bug"

# 路径 3(生成过短 -> teacher-forced):gen_ids 只回 1 个
class ShortClient(FakeClient):
    def rollout(self, prompt, **kw):
        return [9000], PACKED.clone()


loader = RolloutPrefetchLoader([], endpoints=["http://x"], n_target_layers=N_LAYERS,
                               max_length=10000, max_new_tokens=512)
loader.client = ShortClient()
seq3, lm3, th3, tl3 = loader._rollout_one(ids.clone(), lm.clone())
assert torch.equal(seq3, ids) and torch.equal(lm3, lm), "回退路径应保留原文本与 loss_mask"
assert bool((th3.float() != 999).all()), "回退路径 prefill 裁剪 bug"

print("结果:", "三条路径全部通过 ✓" if ok else "不一致 ✗")
sys.exit(0 if ok else 1)
