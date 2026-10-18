"""对照 scipy 验证 pytorch3d shim 的正确性。

这个测试是整个 Windows 方案的关键防线。旋转约定写错**不会抛异常**，只会让
GVHMR 静默输出"看起来合理但错误"的结果——正是我们要避免的失败模式。

scipy.spatial.transform.Rotation 是本领域久经考验的实现，作为独立参照。
注意它用 (x, y, z, w) 四元数（实部在后），而 pytorch3d 用 (w, x, y, z)（实部在前）。

运行：  python tests/test_pytorch3d_shim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shims"))

import pytorch3d  # noqa: E402
from pytorch3d import transforms as T  # noqa: E402
from pytorch3d.ops import knn  # noqa: E402

from scipy.spatial.transform import Rotation as R_scipy  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def t(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def nonidentity_rotation() -> np.ndarray:
    """90° 绕 Z 轴。选它是因为行/列约定在这上面会给出不同答案，能真正区分。"""
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
print("\n[1] 约定验证（常数用例，最能暴露约定搞错）")
# ---------------------------------------------------------------------------
R = t(nonidentity_rotation())

q = T.matrix_to_quaternion(R).numpy()
# 90°绕Z: (w,x,y,z) = (cos45, 0, 0, sin45)
expect_q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
check("matrix_to_quaternion 返回 (w,x,y,z) 顺序",
      np.allclose(q, expect_q, atol=1e-9), f"得到 {np.round(q,6)}")

d6 = T.matrix_to_rotation_6d(R).numpy()
# pytorch3d 取前两 **行**： [R00,R01,R02, R10,R11,R12] = [0,-1,0, 1,0,0]
check("matrix_to_rotation_6d 取前两行（非前两列）",
      np.allclose(d6, [0, -1, 0, 1, 0, 0], atol=1e-9), f"得到 {np.round(d6,6)}")

m = T.quaternion_to_matrix(t([1.0, 0, 0, 0])).numpy()
check("单位四元数 -> 单位矩阵", np.allclose(m, np.eye(3), atol=1e-9))

# ---------------------------------------------------------------------------
print("\n[2] 往返一致性")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
Rs = R_scipy.random(200, random_state=1).as_matrix()

Rt = t(Rs)
check("matrix->quat->matrix 往返", np.allclose(
    T.quaternion_to_matrix(T.matrix_to_quaternion(Rt)).numpy(), Rs, atol=1e-9))
check("matrix->6d->matrix 往返", np.allclose(
    T.rotation_6d_to_matrix(T.matrix_to_rotation_6d(Rt)).numpy(), Rs, atol=1e-9))

aa = rng.normal(size=(200, 3))
check("axis_angle->matrix->axis_angle 往返（比矩阵）", np.allclose(
    T.axis_angle_to_matrix(T.matrix_to_axis_angle(T.axis_angle_to_matrix(t(aa)))).numpy(),
    T.axis_angle_to_matrix(t(aa)).numpy(), atol=1e-8))

# ---------------------------------------------------------------------------
print("\n[3] 对照 scipy（随机旋转，含大角度）")
# ---------------------------------------------------------------------------
for label, rots in (
    ("小角度 (<30°)", rng.normal(size=(100, 3)) * 0.3),
    ("中角度 (<120°)", rng.normal(size=(100, 3)) * 1.0),
    ("近 180°", np.concatenate([rng.normal(size=(50, 3)) * 3.0,
                                R_scipy.random(50, random_state=7).as_rotvec()])),
):
    got = T.axis_angle_to_matrix(t(rots)).numpy()
    want = R_scipy.from_rotvec(rots).as_matrix()
    err = np.abs(got - want).max()
    check(f"axis_angle_to_matrix vs scipy [{label}]", err < 1e-8, f"最大误差 {err:.2e}")

    # matrix_to_quaternion 的验证分三层，因为同一旋转的四元数有 ± 双覆盖
    Rt2 = t(want)
    got_q = T.matrix_to_quaternion(Rt2).numpy()      # (w, x, y, z)

    # (a) 它代表的旋转是对的（这才是唯一真正重要的性质）
    err = np.abs(T.quaternion_to_matrix(t(got_q)).numpy() - want).max()
    check(f"matrix_to_quaternion 旋转正确 vs scipy [{label}]", err < 1e-12, f"最大误差 {err:.2e}")

    # (b) 规范化性质：绝对值最大的那个分量为正
    #     （pytorch3d 的实际约定是"最大分量为正"，不是"w>=0"）
    amax = np.abs(got_q).argmax(axis=1)
    check(f"matrix_to_quaternion 最大分量为正 [{label}]",
          bool((got_q[np.arange(len(got_q)), amax] >= 0).all()))

    # (c) 与 scipy 逐分量比，容许整体 ± 符号后应完全一致
    want_q = R_scipy.from_matrix(want).as_quat()     # (x,y,z,w)
    want_wxyz = np.concatenate([want_q[:, 3:], want_q[:, :3]], axis=1)
    d = np.minimum(np.abs(got_q - want_wxyz).max(axis=1),
                   np.abs(got_q + want_wxyz).max(axis=1))
    check(f"matrix_to_quaternion vs scipy（容许 ±）[{label}]", d.max() < 1e-7,
          f"最大误差 {d.max():.2e}")

    # axis-angle 有符号/分支歧义，比它们代表的旋转而不是向量
    got_aa = T.matrix_to_axis_angle(Rt2)
    err = np.abs(T.axis_angle_to_matrix(got_aa).numpy() - want).max()
    check(f"matrix_to_axis_angle vs scipy [{label}]（比旋转）", err < 1e-8, f"最大误差 {err:.2e}")

# ---------------------------------------------------------------------------
print("\n[4] so3_exp_map / so3_log_map 与 axis_angle 一致")
# ---------------------------------------------------------------------------
check("so3_exp_map == axis_angle_to_matrix",
      np.allclose(T.so3_exp_map(t(aa)).numpy(), T.axis_angle_to_matrix(t(aa)).numpy(), atol=1e-12))
check("so3_log_map 与 matrix_to_axis_angle 代表同一旋转",
      np.allclose(T.axis_angle_to_matrix(T.so3_log_map(Rt)).numpy(),
                  Rs, atol=1e-8))
check("零向量 -> 单位矩阵（小角泰勒分支）",
      np.allclose(T.axis_angle_to_matrix(t([[0.0, 0, 0]])).numpy(), np.eye(3), atol=1e-12))
check("极小角度数值稳定",
      np.isfinite(T.axis_angle_to_matrix(t([[1e-9, 0, 0]])).numpy()).all())

# ---------------------------------------------------------------------------
print("\n[5] 欧拉角（GVHMR 只用到 YXZ 且仅首分量非零）")
# ---------------------------------------------------------------------------
ang = t([[0.7, 0.0, 0.0]])
got = T.euler_angles_to_matrix(ang, "YXZ").numpy()[0]
want = R_scipy.from_euler("Y", 0.7).as_matrix()
check("euler_angles_to_matrix(_, 'YXZ') 首分量 = 绕 Y", np.allclose(got, want, atol=1e-12))

# 三轴都有值时，与 scipy 的同一约定比对
ea = rng.normal(size=(50, 3))
got = T.euler_angles_to_matrix(t(ea), "YXZ").numpy()
want = R_scipy.from_euler("YXZ", ea).as_matrix()
check("三轴欧拉角 vs scipy 'YXZ'", np.abs(got - want).max() < 1e-9,
      f"最大误差 {np.abs(got-want).max():.2e}")

# ---------------------------------------------------------------------------
print("\n[6] knn_points")
# ---------------------------------------------------------------------------
p1 = t(rng.normal(size=(1, 40, 3)))
p2 = t(rng.normal(size=(1, 300, 3)))
d, idx, nn = knn.knn_points(p1, p2, K=1, return_nn=False)
brute = torch.cdist(p1, p2).pow(2).min(dim=-1).values
check("knn_points 返回平方距离且与暴力解一致",
      np.allclose(d[..., 0].numpy(), brute.numpy(), atol=1e-10))
check("return_nn=False 时不返回近邻", nn.numel() == 0)

_, idx3, nn3 = knn.knn_points(p1, p2, K=3, return_nn=True)
check("K=3 近邻坐标 gather 正确",
      np.allclose(nn3.numpy(), p2[0, idx3[0].numpy(), :].numpy(), atol=1e-10))

# ---------------------------------------------------------------------------
print("\n[7] 渲染器桩应明确报错（而非静默返回假数据）")
# ---------------------------------------------------------------------------
try:
    from pytorch3d.renderer import MeshRenderer
    MeshRenderer()
    check("renderer 桩抛错", False, "竟然没抛错")
except NotImplementedError:
    check("renderer 桩抛错", True)
except Exception as e:
    check("renderer 桩抛错", False, f"抛了 {type(e).__name__} 而非 NotImplementedError")

# ---------------------------------------------------------------------------
# Windows 控制台默认 GBK，下面的 ✓/✗ 会抛 UnicodeEncodeError——断言全过却崩在打印上
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项失败: {FAILED}")
    sys.exit(1)
print("✓ 全部通过 —— shim 与 scipy 一致，可安全用于 GVHMR 推理")
