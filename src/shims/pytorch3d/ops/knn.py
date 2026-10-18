"""pytorch3d.ops.knn 的纯 PyTorch 替代。

GVHMR 只用到一处：hmr4d/utils/geo_transform.py 里的
    squared_dist, _, _ = knn.knn_points(points[None], query_verts[None], K=1, return_nn=False)
即"对 p1 的每个点，在 p2 中找 K 个最近邻，返回平方距离"。用 cdist 实现即可。

注意 pytorch3d 返回的是**平方**距离，不是距离本身。
"""

from __future__ import annotations

import torch

__all__ = ["knn_points", "knn_gather"]


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    K: int = 1,
    return_nn: bool = True,
    return_sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """在 p2 中为 p1 的每个点找 K 个最近邻。

    Args:
        p1: (N, P1, D) 查询点
        p2: (N, P2, D) 参考点
        K:  近邻个数
    Returns:
        (dists, idx, nn)
        dists: (N, P1, K) **平方**距离
        idx:   (N, P1, K) 在 p2 中的索引
        nn:    (N, P1, K, D) 近邻点本身；return_nn=False 时为空张量
    """
    if p1.dim() != 3 or p2.dim() != 3:
        raise ValueError(f"knn_points 需要 (N, P, D) 输入，收到 {tuple(p1.shape)} 和 {tuple(p2.shape)}")
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("p1 与 p2 的 batch 维度必须一致")
    if K > p2.shape[1]:
        raise ValueError(f"K={K} 超过参考点数量 {p2.shape[1]}")

    # (N, P1, P2) 平方欧氏距离
    dists_full = torch.cdist(p1, p2, p=2).pow(2)
    dists, idx = dists_full.topk(K, dim=-1, largest=False, sorted=return_sorted)

    if not return_nn:
        return dists, idx, torch.empty(0, dtype=torch.int64, device=p1.device)

    # gather 出近邻坐标
    nn = torch.gather(
        p2.unsqueeze(1).expand(-1, p1.shape[1], -1, -1),
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, p2.shape[-1]),
    )
    return dists, idx, nn


def knn_gather(idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """按 idx 从 x 中取值。idx: (N, P, K)，x: (N, Q, D) -> (N, P, K, D)。"""
    return torch.gather(
        x.unsqueeze(1).expand(-1, idx.shape[1], -1, -1),
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]),
    )
