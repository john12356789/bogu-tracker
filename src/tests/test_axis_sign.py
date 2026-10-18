"""六个轴的正方向：符号约定必须钉死——错了不会崩，只会"机器反着动"。

    python src/tests/test_axis_sign.py

为什么单独一个文件：旋转轴的符号在这条链上有三处会互相影响——关节构基
（pelvis_frames）→ euler 取哪一位 → 要不要取负。**取负那一步出过错**：R2 原来
写的是 `-eul[:, iX]`（注释说"L2 的正方向是使用者左侧，所以绕左 = -绕右"），方向
是反的。2026-10-17 用真素材独立量出来（BV1RZY4zPEw2，世界系）：

    corr(R2_raw, 骨盆前倾角) = -0.66；R2 最大的 10% 帧平均前倾 +4.4°、
    最小的 10% 帧 +19.2°   →  R2 越大反而越往后仰

翻转之后 corr = +0.66、R2 最大 → 前倾 19.2°。这里用**合成的关节**把这条链钉住
（不需要任何素材）：把骨盆向前倾一个已知角度，断言 R2 为正。

中立位取**首帧**（与 --neutral first 一致），所以每个用例都是"先给一帧直立的，
再给待测姿态"——两帧都倾斜的话相对姿态恒为 0，什么都测不到。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import numpy as np                                    # noqa: E402
from scipy.spatial.transform import Rotation          # noqa: E402
import to_funscript as tf                             # noqa: E402

ORDER = "YZX"
iY, iZ, iX = ORDER.index("Y"), ORDER.index("Z"), ORDER.index("X")
FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def frame(spine_tip=(0.0, 0.2, 0.0), hip_deg: float = 0.0) -> np.ndarray:
    """一帧合成关节 (24,3)：0=pelvis 1=左胯 2=右胯 3=spine1，其余点留零。

    `spine_tip` 往前挪 = 骨盆前倾（前倾是绕局部右轴转，胯线方向不变）；
    `hip_deg` 让胯线绕**前**轴转（模拟侧倾姿态下的胯线）。
    """
    J = np.zeros((24, 3), dtype=np.float64)
    th = np.deg2rad(hip_deg)
    J[1] = (-0.10 * np.cos(th), -0.10 * np.sin(th), 0.0)     # 左胯
    J[2] = (0.10 * np.cos(th), 0.10 * np.sin(th), 0.0)       # 右胯
    J[3] = spine_tip
    return J


def angles(pose: np.ndarray, order: str = ORDER) -> np.ndarray:
    """[直立, 待测] 两帧 -> 逐帧欧拉角（管线里那条链：构基→以首帧为中立→euler）。"""
    J = np.stack([frame(), pose])
    M = tf.pelvis_frames(J)
    M0 = tf.orthonormal(M[0])
    R = np.stack([M0.T @ m for m in M])
    return Rotation.from_matrix(R).as_euler(order, degrees=True)


def main() -> int:
    print("\n[1] 直立时三个角都是 0")
    e = angles(frame())
    check(np.all(np.abs(e[1]) < 1e-6), f"直立 -> {np.round(e[1], 3).tolist()}")

    print("\n[2] R2（俯仰）：骨盆**向前倾**为正、向后仰为负")
    for deg in (20.0, -20.0):
        th = np.deg2rad(abs(deg))
        tip = (0.0, 0.2 * np.cos(th), np.sign(deg) * 0.2 * np.sin(th))
        r2 = angles(frame(spine_tip=tip))[1, iX]
        check(np.sign(r2) == np.sign(deg) and abs(r2) > 0.9 * abs(deg),
              f"骨盆{'前倾' if deg > 0 else '后仰'} {abs(deg):.0f}° -> R2 = {r2:+.1f}°")

    print("\n[3] 另外两条轴的当前约定（**未与设备核对**，只记录实现）")
    th = np.deg2rad(30.0)
    spin = frame()
    spin[1] = (-0.10 * np.cos(th), 0.0, 0.10 * np.sin(th))     # 左胯往前
    spin[2] = (0.10 * np.cos(th), 0.0, -0.10 * np.sin(th))     # 右胯往后
    e_spin = angles(spin)[1]
    print(f"     绕上轴 30°（左胯往前）-> R0 = {e_spin[iY]:+.1f}°  "
          f"R1 = {e_spin[iZ]:+.1f}°  R2 = {e_spin[iX]:+.1f}°")
    e_roll = angles(frame(hip_deg=20.0))[1]
    print(f"     绕前轴 20°（右胯抬起）-> R1 = {e_roll[iZ]:+.1f}°  "
          f"R2 = {e_roll[iX]:+.1f}°")
    check(abs(e_spin[iY]) > 20, f"转身只落在 R0 上（{e_spin[iY]:+.1f}°）")

    print()
    if FAILS:
        print(f"✗ {len(FAILS)} 项没过：")
        for f in FAILS:
            print("   -", f)
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
