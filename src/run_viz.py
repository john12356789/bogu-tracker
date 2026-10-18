"""骨骼追踪可视化入口。

用法
----
  # 合成 demo，自检框架（模型还在下载时用这个）
  python run_viz.py --demo

  # demo + 跨方法分歧演示（造一个"第二个方法"来看分歧视图长什么样）
  python run_viz.py --demo --compare

  # 真实视频 + 重建结果
  python run_viz.py --video "你的视频.mp4" --npz result.npz

  # 只导出 .rrd，稍后用 `rerun out.rrd` 打开（不启动查看器）
  python run_viz.py --demo --save out.rrd

  # 调整 demo 的骨盆遮挡区间
  python run_viz.py --demo --occlude 60 120
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="骨骼追踪可视化与诊断（Rerun）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="使用合成运动数据（默认）")
    src.add_argument("--npz", metavar="PATH", help="从 npz 结果文件读取重建结果")

    p.add_argument("--video", metavar="PATH", help="真实视频路径，用于视频面板")
    p.add_argument("--compare", action="store_true", help="演示跨方法分歧（demo 模式下造一个伪方法）")
    p.add_argument("--save", metavar="PATH", help="导出 .rrd 而不是启动查看器")
    p.add_argument("--frames", type=int, default=240, help="demo 帧数（默认 240）")
    p.add_argument("--fps", type=float, default=30.0, help="demo 帧率（默认 30）")
    p.add_argument("--occlude", nargs=2, type=int, metavar=("START", "END"),
                   help="demo 中骨盆被遮挡的帧区间")
    p.add_argument("--stride", type=int, default=1, help="视频抽帧步长（默认 1）")
    p.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧")
    p.add_argument("--no-video", action="store_true", help="不显示视频面板")
    return p


def make_pseudo_second_method(seq, occlude_range):
    """造一个"第二个方法"：可见段贴合、遮挡段漂移。

    用来演示分歧视图。真实使用时应换成 WHAM/GVHMR 的另一份输出——
    两者在遮挡段是否岔开，就是你的不确定度。
    """
    from viz.schema import FOOT_JOINT_IDS, MotionSequence

    rng = np.random.default_rng(7)
    joints = seq.joints.copy()
    n = seq.n_frames
    ramp = np.zeros(n)
    ramp[occlude_range[0]: occlude_range[1]] = np.linspace(
        0, 1, max(occlude_range[1] - occlude_range[0], 1))
    ramp[occlude_range[1]:] = 1.0
    taper = np.clip(1.0 - (np.arange(n) - occlude_range[1]) / 40.0, 0.0, 1.0)
    drift = ramp * np.maximum(taper, 0.0)
    joints[:, 0, 0] += 0.11 * drift
    joints[:, 0, 2] += 0.09 * drift
    joints[:, 0, 1] += 0.03 * drift
    joints += rng.normal(0, 0.004, size=joints.shape)
    joints[:, FOOT_JOINT_IDS, :] = seq.joints[:, FOOT_JOINT_IDS, :]

    return MotionSequence(
        name="pseudo-method-B", fps=seq.fps, joints=joints, K=seq.K,
        cam_to_world=seq.cam_to_world, image_size=seq.image_size,
        keypoints_visible=seq.keypoints_visible, source="pseudo",
    )


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认不是 UTF-8，诊断报告里的中文会变乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    args = build_parser().parse_args(argv)

    try:
        import rerun as rr
    except ImportError:
        print("缺少 rerun-sdk。请先运行：  pip install rerun-sdk", file=sys.stderr)
        return 1

    from viz import app, checks, sources

    occlude = tuple(args.occlude) if args.occlude else (90, 150)

    # ---- 数据源 ----
    real_video = None
    if args.npz:
        seq = sources.from_smplx_npz(args.npz, fps=args.fps)
    elif args.demo:
        seq = sources.make_demo(n_frames=args.frames, fps=args.fps, occlude_range=occlude)
    else:
        print("未指定数据源，使用 --demo。", file=sys.stderr)
        seq = sources.make_demo(n_frames=args.frames, fps=args.fps, occlude_range=occlude)

    if args.max_frames:
        seq = _truncate(seq, args.max_frames)

    # ---- 视频面板 ----
    video_frames = None
    if not args.no_video:
        if args.video:
            try:
                real_video, size, vid_fps = sources.load_video_frames(
                    args.video, max_frames=args.max_frames, stride=args.stride)
                video_frames = real_video
                print(f"已载入视频：{len(real_video)} 帧，{size[0]}x{size[1]} @ {vid_fps:.2f} fps")
                if args.demo:
                    print("  注意：demo 骨架与真实视频不对应，视频面板仅用于确认读取正常。")
                    print("        要叠加对齐，请用 --npz 提供该视频的重建结果。")
            except Exception as exc:
                print(f"视频读取失败，跳过视频面板：{exc}", file=sys.stderr)
        elif args.demo:
            video_frames = sources.render_demo_frames(seq, stride=args.stride)

    # ---- 诊断 ----
    ghost = None
    if args.compare:
        ghost = make_pseudo_second_method(seq, occlude)

    report, div = checks.run_all(seq, other=ghost)

    print(f"\n序列：{seq.name}   {seq.n_frames} 帧 @ {seq.fps:.1f} fps   来源：{seq.source}")
    print("-" * 62)
    print(report.summary())
    if div is not None:
        print(f"跨方法分歧   均值 {div.mean():.4f} m，峰值 {div.max():.4f} m"
              f"（峰值出现在第 {int(np.argmax(div))} 帧）")
    print("-" * 62)

    # ---- 可视化 ----
    # app_id 必须是 ASCII，否则 Rerun 会做 entry name 迁移并改名
    safe_name = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "-" for c in seq.name)
    rr.init(f"skeleton-tracking-{safe_name}", spawn=args.save is None)
    if args.save:
        rr.save(args.save)

    app.send_blueprint(rr, has_video=bool(video_frames), has_metrics=True, has_ghost=ghost is not None)
    app.log_sequence(rr, seq, report, video_frames=video_frames,
                     video_stride=args.stride, divergence=div, ghost=ghost)

    if args.save:
        print(f"已写入 {args.save}\n用 `rerun {args.save}` 打开。")
    else:
        print("查看器已启动。")
        print("提醒：3D 视图在遮挡段会看起来'很对'——那是它被设计成这样的。")
        print("      判断对错请看重投影误差曲线和跨方法分歧。")
    return 0


def _truncate(seq, n):
    from viz.schema import MotionSequence

    fields = {}
    for key in ("vertices", "joints", "keypoints_2d", "keypoints_conf",
                "keypoints_visible", "foot_contact"):
        val = getattr(seq, key)
        fields[key] = None if val is None else np.asarray(val)[:n]
    return MotionSequence(
        name=seq.name, fps=seq.fps, faces=seq.faces, K=seq.K,
        cam_to_world=seq.cam_to_world, image_size=seq.image_size,
        ground_y=seq.ground_y, joint_names=seq.joint_names, source=seq.source,
        kp_to_joint=seq.kp_to_joint, **fields,
    )


if __name__ == "__main__":
    raise SystemExit(main())
