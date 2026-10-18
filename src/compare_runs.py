"""并排对比多次重建的诊断指标。

动机
----
`viz/checks.py` 算的是**单次**重建的诊断。要判断「60fps 比 30fps 好还是差」这类
问题，需要把两次运行放在同一张表里，并且**把跟帧率耦合的量换算成物理单位**——
否则会得出错误结论。

两个必须换算的量（最容易踩的坑）
--------------------------------
  root_jitter  是骨盆位置的**二阶差分**（m/frame²），没除 dt²。
               60fps 的 dt 是 30fps 的一半，二阶差分随之小 4 倍。
               直接比会看到「60fps 抖动小 4 倍」这种假象。乘 fps² 得 m/s² 才能比。
  foot_slide   是**逐帧**水平位移（m/frame）。60fps 每帧走一半距离。
               乘 fps 得 m/s 才能比。

重投影误差（像素）与帧率无关，可以直接比。

用法
----
    python src/compare_runs.py A_incam.npz B_incam.npz --fps 30 60 --labels 30fps 60fps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from viz import checks, sources  # noqa: E402
from viz.schema import COCO17_KEYPOINT_NAMES  # noqa: E402

# COCO17 里左右胯的下标，用户关心的区域
HIP_KP = (11, 12)


def load(path: str, fps: float) -> "sources.MotionSequence":
    return sources.from_smplx_npz(path, fps=fps, name=Path(path).parent.name)


def phys_jitter(seq, report) -> float:
    """骨盆抖动换算成 m/s²（乘 fps²）。"""
    return float(np.median(report.jitter) * seq.fps ** 2)


def phys_slide(seq, report) -> float:
    """足滑换算成 m/s（乘 fps）。"""
    s = report.slide[report.slide > 0]
    return float(s.mean() * seq.fps) if s.size else 0.0


def hip_reproj(report) -> float:
    if report.reproj_per_kp is None:
        return float("nan")
    vals = [report.reproj_per_kp[i] for i in HIP_KP if i < len(report.reproj_per_kp)]
    return float(np.nanmedian(vals))


def all_reproj(report) -> float:
    if report.reproj_per_kp is None:
        return float("nan")
    return float(np.nanmedian(report.reproj_per_kp))


def pelvis_world(seq) -> np.ndarray:
    """骨盆世界轨迹 (T,3)。"""
    return np.asarray(seq.joints)[:, 0, :]


def resample_track(track: np.ndarray, fps: float, t_grid: np.ndarray) -> np.ndarray:
    """按**时间**重采样轨迹到统一时间轴。

    帧数不同只能按时间对齐——60fps 抽一半帧不等于 30fps（曝光/运动模糊不同），
    所以用插值到共同时间网格，而不是直接隔帧取。
    """
    t_src = np.arange(len(track)) / fps
    return np.stack([np.interp(t_grid, t_src, track[:, k]) for k in range(3)], axis=1)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="并排对比多次重建的诊断指标")
    ap.add_argument("npz", nargs="+", help="各次运行的 *_incam.npz")
    ap.add_argument("--fps", type=float, nargs="+", required=True,
                    help="与 npz 一一对应的帧率（npz 里存的 fps 字段不可信）")
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()

    if len(args.fps) != len(args.npz):
        print("--fps 个数必须与 npz 个数相同", file=sys.stderr)
        return 1
    labels = args.labels or [Path(p).parent.name for p in args.npz]
    if len(labels) != len(args.npz):
        print("--labels 个数必须与 npz 个数相同", file=sys.stderr)
        return 1

    seqs, reports = [], []
    for path, fps in zip(args.npz, args.fps):
        seq = load(path, fps)
        report, _ = checks.run_all(seq)
        seqs.append(seq)
        reports.append(report)

    # ---- 主表 ----
    rows = [
        ("帧数", lambda q, r: f"{q.n_frames}", "整数"),
        ("时长 (s)", lambda q, r: f"{q.n_frames / q.fps:.2f}", "s"),
        ("重投影误差 中位", lambda q, r: f"{all_reproj(r):.2f}", "px  ← 与帧率无关，可直接比"),
        ("重投影误差 胯部中位", lambda q, r: f"{hip_reproj(r):.2f}", "px  ← 你关心的区域"),
        ("骨盆抖动 中位", lambda q, r: f"{phys_jitter(q, r):.2f}", "m/s² ← 已乘 fps²，可跨帧率比"),
        ("足滑 均值", lambda q, r: f"{phys_slide(q, r):.3f}", "m/s  ← 已乘 fps，可跨帧率比"),
        ("骨盆世界轨迹长度", lambda q, r: f"{_path_len(pelvis_world(q)):.3f}", "m"),
    ]

    w = max(len(r[0]) for r in rows) + 2
    col = max(16, max(len(l) for l in labels) + 2)
    print()
    print(" " * w + "".join(f"{l:>{col}s}" for l in labels))
    print("-" * (w + col * len(labels)))
    for name, fn, unit in rows:
        cells = "".join(f"{fn(q, r):>{col}s}" for q, r in zip(seqs, reports))
        print(f"{name:<{w}s}{cells}   {unit}")

    # ---- 逐关节重投影误差 ----
    print()
    print("逐关节重投影误差中位 (px)")
    print("-" * (w + col * len(labels)))
    k = min(len(r.reproj_per_kp) for r in reports if r.reproj_per_kp is not None)
    print(f"{'关键点':<{w}s}" + "".join(f"{l:>{col}s}" for l in labels))
    for i in range(k):
        name = COCO17_KEYPOINT_NAMES[i] if i < len(COCO17_KEYPOINT_NAMES) else f"kp{i}"
        vals = [r.reproj_per_kp[i] for r in reports]
        star = "  ★" if i in HIP_KP else ""
        print(f"{name:<{w}s}" + "".join(f"{v:>{col}.2f}" for v in vals) + star)

    # ---- 世界系骨盆轨迹一致性 ----
    if len(seqs) >= 2:
        print()
        print("世界系骨盆轨迹一致性（按时间重采样到共同时间轴后比）")
        print("-" * 66)
        t_max = min(q.n_frames / q.fps for q in seqs)
        t_grid = np.linspace(0.0, t_max, 200)
        ref = resample_track(pelvis_world(seqs[0]), seqs[0].fps, t_grid)
        for lab, q in zip(labels[1:], seqs[1:]):
            cur = resample_track(pelvis_world(q), q.fps, t_grid)
            d = cur - ref
            # 注意是 RMS（先平方求均值再开方），不是 np.linalg.norm(d, axis=0)——
            # 后者是沿时间轴求 L2 范数，会随采样点数增长而虚增约 sqrt(N) 倍。
            per_axis = np.sqrt((d ** 2).mean(axis=0))
            print(f"  {labels[0]} vs {lab}")
            print(f"     逐轴 RMS   X {per_axis[0]*100:6.2f} cm   "
                  f"Y {per_axis[1]*100:6.2f} cm   Z {per_axis[2]*100:6.2f} cm")
            corr = [np.corrcoef(cur[:, k], ref[:, k])[0, 1] for k in range(3)]
            print(f"     逐轴相关   X {corr[0]:6.3f}      Y {corr[1]:6.3f}      Z {corr[2]:6.3f}")
            step = np.linalg.norm(np.diff(d, axis=0), axis=1)
            print(f"     逐帧距离   中位 {np.median(np.linalg.norm(d, axis=1))*100:.2f} cm，"
                  f"峰值 {np.linalg.norm(d, axis=1).max()*100:.2f} cm")
            print(f"     距离振荡   中位 {np.median(step)*100:.3f} cm/步，峰值 {step.max()*100:.3f}"
                  "   ← 时变振荡说明轨迹是各自漂移的，不是常量偏移")
    print()
    return 0


def _path_len(track: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())


if __name__ == "__main__":
    raise SystemExit(main())
