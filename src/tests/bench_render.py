"""量渲染的并行效率：纯渲染吞吐（不编码）vs worker 数。

为什么单列「纯渲染」这一栏
--------------------------
端到端时间 = 渲染/并行效率 + 编码。x264 是单个进程、不随 worker 数并行，
所以端到端倍数必然低于纯渲染倍数。把两者分开量，才能判断剩下的损耗
是「渲染没并行好」还是「编码占了固定比例」。
（实测 quad 1119 帧：端到端 240s→85s 只有 2.8×，而编码约占 20%，
 理论上限就是 5×——所以要看纯渲染能不能到 6× 以上。）

**必须作为文件运行，不能 `python - <<EOF`**：Windows 用 spawn 起子进程，
子进程要按 __main__.__file__ 重新导入主模块，stdin 脚本的 __file__ 是
`<stdin>`，子进程会直接 OSError。

用法
----
    python src/tests/bench_render.py                 # 默认 300 帧、1/4/8/12 worker
    python src/tests/bench_render.py --frames 100 --workers 1 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import render_overlay as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "BV1eyYt6QEwJ_60"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="渲染并行效率基准")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="outputs 下的目录名")
    ap.add_argument("--start", type=int, default=0, help="从第几帧开始（用于切片对照）")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 12])
    ap.add_argument("--layout", default="quad")
    args = ap.parse_args()

    stem = f"{args.tag}_1280p"
    d = ROOT / "workspace/outputs" / args.tag / "gvhmr_rtmpose" / stem
    video = ROOT / "workspace/processed" / f"{stem}.mp4"
    if not (d / "hmr4d_results_incam.npz").exists():
        print(f"缺少 {d/'hmr4d_results_incam.npz'}", file=sys.stderr)
        return 1

    lo = np.load(d / "hmr4d_results_incam.npz")
    glo = np.load(d / "hmr4d_results_global.npz")
    A = {"verts": lo["vertices"], "joints": lo["joints"]}
    K = lo["K"]
    Ks = np.tile(K, (len(lo["vertices"]), 1, 1)) if K.ndim == 2 else K
    gv, gj = glo["vertices"], glo["joints"]

    cap = cv2.VideoCapture(str(video))
    frames = []
    idx = 0
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        if idx >= args.start:
            frames.append(f)
        idx += 1
    cap.release()
    n = min(len(frames), len(A["verts"]))
    h, w = frames[0].shape[:2]
    panels = {"overlay": 1, "sidebyside": 2, "triple": 3, "quad": 4}[args.layout]
    out_w = panels * w

    ctx = {
        "faces": np.asarray(lo["faces"]), "layout": args.layout, "panels": panels,
        "alpha": 0.55, "w": w, "h": h,
        "cam_facing": np.array([0.0, 0.0, 1.0]),
        "cam_side": np.array([1.0, 0.0, 0.0]),
        "grid_center": (0.0, 0.0), "grid_step": 0.5, "grid_extent": 4.0,
    }

    def pack(i):
        j = args.start + i
        return [frames[i], A["verts"][j], A["joints"][j], Ks[j], gv[j], gj[j]]

    print(f"{args.tag}  {n} 帧 {args.layout}  {out_w}x{h}   纯渲染吞吐（sink 丢弃，不含编码）\n")
    base = None
    for wk in args.workers:
        if wk == 1:
            build = R.make_frame_builder(ctx)
            t = time.time()
            for i in range(n):
                build(pack(i))
        else:
            t = time.time()
            R.render_parallel(n, pack, ctx, (h, out_w, 3), wk, lambda b: None, out_w, h)
        dt = time.time() - t
        if base is None:
            base = dt
        print(f"  workers={wk:2d}  {dt:7.1f}s  {dt/n:6.4f} s/帧   {base/dt:4.1f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
