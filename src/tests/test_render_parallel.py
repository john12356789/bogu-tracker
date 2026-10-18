"""验证多进程渲染与单进程**逐字节等价**。

为什么不用「渲染两个 mp4 再比文件」
----------------------------------
那样要对同一段视频渲染两遍，慢且没必要。关键在于：单进程和多进程调的是
**同一个** `make_frame_builder(ctx)` 返回的 `build()`。所以两者唯一的差别
只可能是**帧序**——并行时帧是乱序完成的，得靠主进程重排后按序写出。

所以这里分两个测试：
  1) 帧序测试：用极小的合成数据（32x24 画布、1 个三角形），每帧的像素值
     编码帧号，让 build 几乎零成本。200 帧、8 个 worker 跑一遍，
     断言 sink 收到的缓冲区**顺序与内容**都正确。这是真正有风险的部分，毫秒级完成。
  2) 真实数据冒烟测试：用真实 npz + 视频跑 4 帧，逐字节比对单进程与多进程的输出。
     确认真实路径在 worker 里也能跑通。

用法
----
    python src/tests/test_render_parallel.py
    python src/tests/test_render_parallel.py --real-frames 8 --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import render_overlay as R  # noqa: E402
from mesh_npz import mesh_path_for  # noqa: E402


def base_ctx(w, h, layout, panels, faces):
    # `cells` / `grid` / `cell_h` 是 make_frame_builder 真正读的三个键
    # （main 里在 ctx 建成之前算好：`grid = parse_grid(...)`、`cell_h = args.cell_h or h`、
    # LAYOUT_VIEWS 展开成 cells）。这里照着 main 的算法给同一套值，**用真函数**——
    # 自己手写一份 cells 就会和 LAYOUT_VIEWS 漂。
    spec = R.LAYOUT_VIEWS.get(layout, layout)      # quad -> 4 格；overlay 本身就是一格
    cells = R.parse_views(",".join(spec) if isinstance(spec, list) else spec)
    return {
        "faces": faces, "layout": layout, "panels": panels, "alpha": 0.55,
        "cells": cells,
        "grid": None, "cell_h": h,
        "w": w, "h": h,
        "cam_facing": np.array([0.0, 0.0, 1.0]),
        "cam_side": np.array([1.0, 0.0, 0.0]),
        "grid_center": (0.0, 0.0), "grid_step": 0.5, "grid_extent": 4.0,
    }


def run_serial(build, pack, n):
    return [build(pack(i)).tobytes() for i in range(n)]


def run_parallel(build_ctx, pack, n, workers, shape):
    got: list[bytes] = []
    R.render_parallel(n, pack, build_ctx, shape, workers, lambda buf: got.append(bytes(buf)),
                      shape[1], shape[0])
    return got


def test_order(n=200, workers=8):
    """帧序测试：build 几乎零成本，靠像素值编码帧号来暴露任何错序。"""
    w, h = 32, 24
    # 单个三角形，三个顶点都投到画面内
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    ctx = base_ctx(w, h, "overlay", 1, faces)

    def pack(i):
        # `pack` 返回的是**一帧的输入字典**（render_cell 按名字取），不是位置列表。
        frame = np.full((h, w, 3), i % 256, dtype=np.uint8)
        verts = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0]])
        joints = np.zeros((24, 3))
        K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
        return {"frame": frame, "verts": verts, "joints": joints, "K": K,
                "gv": None, "gj": None}

    build = R.make_frame_builder(ctx)
    exp = run_serial(build, pack, n)
    got = run_parallel(ctx, pack, n, workers, (h, w, 3))

    assert len(got) == n, f"帧数不对：收到 {len(got)}，应为 {n}"
    bad = [i for i in range(n) if got[i] != exp[i]]
    if bad:
        raise AssertionError(f"帧序或内容不符，共 {len(bad)} 帧，前几个下标：{bad[:10]}")
    print(f"  [1] 帧序测试通过：{n} 帧 / {workers} worker，顺序与内容逐字节一致")


def test_real(frames_n, workers):
    """真实数据冒烟：单进程 vs 多进程逐字节比对。

    素材原来指的是 `BV1WzeH6REQr`——那条的产物早就删了，于是这个测试一直走
    "跳过"分支、等于没测。换成双人素材的参照物（产物齐、两个坐标系都有）。
    """
    root = Path(__file__).resolve().parents[2]
    d = root / "workspace/outputs/BV1RZY4zPEw2/gvhmr_rtmpose/BV1RZY4zPEw2_1280p_30fps"
    video = root / "workspace/processed/BV1RZY4zPEw2_1280p_30fps.mp4"
    npz = d / "hmr4d_results_incam.npz"
    gnpz = d / "hmr4d_results_global.npz"
    if not npz.exists() or not video.exists():
        print(f"  [2] 跳过真实数据测试（缺少 {npz} 或 {video}）")
        return
    # 顶点现在**单独一个文件**（它占原 npz 的 99.5%，而只有渲染读它，所以平时
    # 不落盘——见 mesh_npz.py）。不在就跳过这一项：渲染任意一条预览会自动补回来，
    # 也可以手敲 export_gvhmr.py 补。
    mesh, gmesh = mesh_path_for(npz), mesh_path_for(gnpz)
    if not mesh.exists() or not gmesh.exists():
        print(f"  [2] 跳过：网格文件不在（{mesh.name}）——渲一次预览会自动补，"
              f"或跑 export_gvhmr.py <hmr4d_results.pt> --spaces incam,global")
        return

    import cv2
    lo = np.load(npz)
    mo = np.load(mesh)
    glo = np.load(gnpz)
    gmo = np.load(gmesh)
    A = {"verts": mo["vertices"], "joints": lo["joints"]}
    K = lo["K"]
    Ks = np.tile(K, (len(mo["vertices"]), 1, 1)) if K.ndim == 2 else K
    gv, gj = gmo["vertices"], glo["joints"]

    cap = cv2.VideoCapture(str(video))
    frames = []
    for _ in range(frames_n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = min(len(frames), len(A["verts"]))
    h, w = frames[0].shape[:2]

    ctx = base_ctx(w, h, "quad", 4, np.asarray(mo["faces"]))

    def pack(i):
        # 一帧的输入字典（render_cell 按名字取）——不是位置列表
        return {"frame": frames[i], "verts": A["verts"][i], "joints": A["joints"][i],
                "K": Ks[i], "gv": gv[i], "gj": gj[i]}

    build = R.make_frame_builder(ctx)
    exp = run_serial(build, pack, n)
    got = run_parallel(ctx, pack, n, workers, (h, 4 * w, 3))

    assert len(got) == n
    bad = [i for i in range(n) if got[i] != exp[i]]
    if bad:
        raise AssertionError(f"真实数据下不一致，帧下标：{bad[:10]}")
    print(f"  [2] 真实数据测试通过：{n} 帧 quad {w}x{h}，单进程与 {workers} 进程输出逐字节一致")


def test_canvas_even() -> None:
    """画布宽高必须是偶数。

    **libx264 配 yuv420p（4:2:0）要求宽高都是偶数**，奇数直接 EINVAL、一个包
    都不写——整段渲染白跑，日志里只有一句 ffmpeg 的报错。实测崩在这里，而画布
    尺寸是"每格等比缩放后的宽度之和 / 槽宽 × 列数"，随手就是奇数。
    """
    import render_overlay as r

    cases = [
        ("一行三格，每格 719 宽（奇数）", [np.zeros((100, 719, 3), np.uint8)] * 3, None, 100),
        ("网格 1x3，同上", [np.zeros((100, 719, 3), np.uint8)] * 3, (1, 3), 100),
        ("奇数 cell_h", [np.zeros((101, 50, 3), np.uint8)], None, 101),
        ("网格 2x2", [np.zeros((100, 719, 3), np.uint8)] * 4, (2, 2), 100),
    ]
    for name, cells, grid, cell_h in cases:
        c = r.assemble(cells, grid, cell_h)
        assert c.shape[0] % 2 == 0 and c.shape[1] % 2 == 0, f"{name}: {c.shape}"
        print(f"  [画布] {name} -> {c.shape[1]}x{c.shape[0]}（都是偶数）")
    # 本来就偶数的**不许**被补：补了会多一条黑边，还会让尺寸随内容漂
    c = r.assemble([np.zeros((100, 720, 3), np.uint8)] * 2, None, 100)
    assert c.shape == (100, 1440, 3), c.shape
    print(f"  [画布] 本来就偶数 -> {c.shape[1]}x{c.shape[0]}（没被补）")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="多进程渲染等价性测试")
    ap.add_argument("--order-frames", type=int, default=200)
    ap.add_argument("--real-frames", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print("多进程渲染等价性")
    test_canvas_even()
    test_order(args.order_frames, args.workers)
    test_real(args.real_frames, args.workers)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
