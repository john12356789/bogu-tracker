"""look_at_rotation 的纯 PyTorch 实现（纯数学，无 CUDA 依赖，直接照搬语义）。

构造"看向 at 点"的相机旋转矩阵：相机 -Z 轴指向 at，+Y 轴尽量对齐 up。
返回世界→相机的旋转 (N, 3, 3)，即矩阵的行是相机基向量在世界的表示。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["look_at_rotation"]


def look_at_rotation(camera_position, at=None, up=None, eps=1e-4, device=None):
    if device is None:
        device = camera_position.device
    if at is None:
        at = torch.zeros_like(camera_position, device=device)
    else:
        at = at.to(device)
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    else:
        up = up.to(device)
        if up.dim() == 1:
            up = up[None].expand(camera_position.shape[0], -1)

    z_axis = F.normalize(at - camera_position, eps=eps)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=eps)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=eps)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)
