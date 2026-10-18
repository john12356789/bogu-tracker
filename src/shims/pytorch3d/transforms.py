"""pytorch3d.transforms 的纯 PyTorch 替代实现。

为什么存在
----------
Meta 只为 pytorch3d 发布了 Linux wheel（已实测：Linux 版 200，Windows 版 403），
GVHMR 的 requirements.txt 直接锁死那个 linux_x86_64 直链。而 GVHMR 推理路径
只用到 transforms 里这些旋转数学 + 一个 knn_points，不需要 CUDA 编译的渲染器。
所以用本模块顶上，就能在原生 Windows 上跑。

约定（与 pytorch3d 一致——这几条写错不会报错，只会静默算错）
-----------------------------------------------------------
1. 四元数顺序是 (w, x, y, z)，**实部在前**。
   （scipy 是 (x, y, z, w)，实部在后——两者互转时必须显式处理。）
2. matrix_to_quaternion 的规范化性质是「**绝对值最大的分量为正**」，
   **不是**「w >= 0」。当最大分量落在 x/y/z 上时，w 可以是负的——
   这是"选最大分量分支"算法的自然结果，也正是 pytorch3d 的实际行为。
   （同一旋转的四元数有 ± 双覆盖，任何比大小/比符号的做法都必须先统一约定。）
3. matrix_to_rotation_6d 取矩阵的**前两行**按行主序展平：
   [R00,R01,R02, R10,R11,R12]。
   注意这**不同于** Zhou et al. 2019 原文（原文取前两列），pytorch3d 用的是行。
4. so3_exp_map 的输入就是轴角本身，旋转角 = 向量模长，不做归一化。
5. matrix_to_axis_angle 走 四元数 路径（比直接取反对称部分在 θ→π 附近数值更稳）。

正确性由 tests/test_pytorch3d_shim.py 对照 scipy.spatial.transform.Rotation 验证。
"""

from __future__ import annotations

import functools

import torch

__all__ = [
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    "so3_exp_map",
    "so3_log_map",
    "euler_angles_to_matrix",
]

_EPS = 1e-6


# ---------------------------------------------------------------------------
# 基础旋转
# ---------------------------------------------------------------------------
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """绕单轴旋转 angle 弧度，返回 (..., 3, 3)。轴由字符 'X'/'Y'/'Z' 指定。"""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    if axis == "X":
        a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2 = (
            one, zero, zero, zero, cos, -sin, zero, sin, cos, zero, zero, one)
    elif axis == "Y":
        a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2 = (
            cos, zero, sin, zero, one, zero, -sin, zero, cos, zero, zero, one)
    elif axis == "Z":
        a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2 = (
            cos, -sin, zero, sin, cos, zero, zero, zero, one, zero, zero, one)
    else:
        raise ValueError(f"Invalid axis {axis!r}, expected one of X/Y/Z.")

    return torch.stack(
        (a0, a1, a2, b0, b1, b2, c0, c1, c2), dim=-1
    ).reshape(angle.shape + (3, 3))


def so3_exp_map(log_rot: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """轴角 (..., 3) -> 旋转矩阵 (..., 3, 3)。等价于 Rodrigues 公式。

    θ 很小时 sinθ/θ 等系数用泰勒展开，避免除零。
    判据用 θ² < eps，与 pytorch3d 一致。
    """
    if log_rot.shape[-1] != 3:
        raise ValueError(f"so3_exp_map 需要最后一维为 3，收到 {tuple(log_rot.shape)}")

    theta2 = (log_rot * log_rot).sum(-1)              # (...,)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    small = (theta2 < eps)

    # sinθ/θ, (1-cosθ)/θ², (θ-sinθ)/θ³ —— 小角处走泰勒
    safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1 - theta2 / 6 + theta2 * theta2 / 120, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - theta2 / 24 + theta2 * theta2 / 720,
                    (1 - torch.cos(safe)) / safe.pow(2))
    c = torch.where(small, 1 / 6 - theta2 / 120 + theta2 * theta2 / 5040,
                    (safe - torch.sin(safe)) / safe.pow(3))

    # K = [log_rot]_×
    x, y, z = log_rot.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(log_rot.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device).expand_as(K)

    return eye + a[..., None, None] * K + b[..., None, None] * (K @ K)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """轴角 (..., 3) -> 旋转矩阵 (..., 3, 3)。"""
    return so3_exp_map(axis_angle)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """四元数 (..., 4) 顺序 (w, x, y, z) -> 旋转矩阵 (..., 3, 3)。

    输入不需要归一化：pytorch3d 用 2/|q|² 因子隐式归一（与手写公式一致）。
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"quaternion_to_matrix 需要最后一维为 4，收到 {tuple(quaternions.shape)}")

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """逐元素开方，负值截到 0（浮点误差可能给出 -1e-9）。"""
    zero = torch.zeros_like(x)
    return torch.sqrt(torch.maximum(x, zero))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 四元数 (..., 4)，顺序 (w, x, y, z)。

    用"选最大分量"分支法，避免 trace 接近 -1（旋转角接近 π）时的数值退化。

    规范化：返回值的**绝对值最大分量为正**（注意不是 w >= 0——见模块文档说明）。
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"matrix_to_quaternion 需要 (..., 3, 3)，收到 {tuple(matrix.shape)}")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        (
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ),
        dim=-2,
    )

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    idx = q_abs.argmax(dim=-1)
    onehot = torch.nn.functional.one_hot(idx, num_classes=4) > 0.5
    return quat_candidates[onehot, :].reshape(batch_dim + (4,))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """轴角 (..., 3) -> 四元数 (..., 4)，(w, x, y, z)。"""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half = angles / 2.0
    small = angles < _EPS
    safe = torch.where(small, torch.ones_like(angles), angles)
    sin_half_over_angle = torch.where(small, 0.5 - angles.pow(2) / 48, torch.sin(half) / safe)
    xyz = axis_angle * sin_half_over_angle
    w = torch.where(small, torch.ones_like(angles), torch.cos(half))
    return torch.cat((w, xyz), dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """四元数 (..., 4)，(w, x, y, z) -> 轴角 (..., 3)，夹角落在 [0, π]。"""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles

    small = angles.abs() < _EPS
    sin_half_over_angle = torch.empty_like(angles)
    sin_half_over_angle[~small] = torch.sin(half_angles[~small]) / angles[~small]
    sin_half_over_angle[small] = 0.5 - angles[small].pow(2) / 48
    return quaternions[..., 1:] / sin_half_over_angle


def so3_log_map(R: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 轴角 (..., 3)，主分支（夹角 ∈ [0, π]）。"""
    return matrix_to_axis_angle(R)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 轴角 (..., 3)。

    走四元数路径：在 θ→π 附近比直接取反对称部分稳定得多。
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


# ---------------------------------------------------------------------------
# 6D 旋转表示
# ---------------------------------------------------------------------------
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 6D (..., 6)。

    **取前两行**按行主序展平： [R00,R01,R02, R10,R11,R12]。
    （Zhou et al. 原文取前两列；pytorch3d 取行。搞混会得到转置的旋转。）
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D (..., 6) -> 旋转矩阵 (..., 3, 3)。Gram-Schmidt 正交化。"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# ---------------------------------------------------------------------------
# 欧拉角
# ---------------------------------------------------------------------------
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """欧拉角 (..., 3)，单位弧度，按 convention 指定的轴序组合。

    convention 形如 "XYZ" / "YXZ"，euler_angles[..., i] 对应 convention[i] 轴。
    """
    if euler_angles.shape[-1] != 3:
        raise ValueError(f"euler_angles_to_matrix 需要最后一维为 3，收到 {tuple(euler_angles.shape)}")
    if len(convention) != 3:
        raise ValueError(f"convention 必须为 3 个字母，收到 {convention!r}")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"无效 convention {convention!r}：相邻轴不能重复")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"convention 含非法字母 {letter!r}")

    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return functools.reduce(torch.matmul, matrices)
