"""用 RTMPose-x 生成 2D 关键点，替代 GVHMR 默认的 ViTPose-H。

动机
----
GVHMR 用的是 ViTPose-H，输入被裁到 **256×192**。这是整条链路 2D 精度的硬天花板。
RTMPose-x 用 **384×288**，像素数是它的 2.25 倍。本脚本产出与 ViTPose 完全同格式的
关键点，让 GVHMR 无感知地换掉这一环，从而做 A/B 对比。

为什么用 rtmlib
---------------
RTMPose 用的是 SimCC 表示（输出两条一维分布而非热图），解码方式特殊且容易写错。
rtmlib 是 MMPose 官方的轻量推理库，且其内置模型文件名与本项目下载的
`rtmpose-x_simcc-body7_pt-body7_700e-384x288-*.onnx` **完全一致**——
说明这些就是 rtmlib 的模型，可直接加载，不必手写解码。

不修改上游仓库
--------------
产物写成一个标准的 `preprocess/vitpose.pt`。GVHMR 的 run_preprocess 里每一步都是
「有产物就跳过」，所以把 bbx.pt 和本脚本产出的 vitpose.pt 预先放好，
再跑 run_gvhmr.py 就只会补跑 HMR2 特征提取和推理。

用法
----
    # 1) 准备一个新输出目录，复用 ViTPose 那次的检测框
    python run_rtmpose.py --src GVHMR/outputs/demo/dance_1280p \
        --dst GVHMR/outputs/demo_rtm --video processed/dance_1280p.mp4

    # 2) 用新目录跑推理（会跳过 YOLO 和 ViTPose）
    python run_gvhmr.py --video processed/dance_1280p.mp4 -s --output_root outputs/demo_rtm
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # src/
ROOT = HERE.parent                              # 项目根
GVHMR_ROOT = ROOT / "GVHMR"
SHIMS = HERE / "shims"
MODELS = ROOT / "workspace" / "models"          # 权重统一放 workspace/models

# 官方 ONNX 文件名里 384x288 是 (高, 宽)；rtmlib 的 model_input_size 要 (宽, 高)
DEFAULT_ONNX = MODELS / "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.onnx"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 RTMPose-x 生成 2D 关键点")
    p.add_argument("--src", required=True, help="ViTPose 那次的输出目录（复用其 bbx.pt）")
    p.add_argument("--dst", required=True, help="新输出目录")
    p.add_argument("--video", required=True, help="视频路径")
    p.add_argument("--onnx", default=str(DEFAULT_ONNX), help="RTMPose ONNX 模型")
    p.add_argument("--device", default="dml", choices=["cuda", "cpu", "dml"],
                   help="3.12 环境的 onnxruntime 没有 CUDA provider，只有 DirectML。"
                        "dml（默认）= 走 DirectML 上 GPU，实测比 CPU 快 6.2 倍"
                        "（21.6 vs 134.6 ms/帧），关键点差异均值 0.0005px、最大 1.84px；"
                        "无 DirectML 时自动回退 cpu。要逐位复现旧结果就传 cpu。"
                        "注意 rtmlib 不认 dml 这个值，见下方手工换 session")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="已存在也重新算")
    return p.parse_args()


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    args = parse_args()
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    video = Path(args.video).resolve()
    onnx = Path(args.onnx).resolve()

    for req, what in ((src / "preprocess/bbx.pt", "检测框"), (video, "视频"), (onnx, "ONNX 模型")):
        if not req.exists():
            print(f"找不到{what}：{req}", file=sys.stderr)
            return 1

    frame_name = src.name                      # 例如 dance_1280p
    out_dir = dst / frame_name
    pre_dir = out_dir / "preprocess"
    dst_vitpose = pre_dir / "vitpose.pt"
    if dst_vitpose.exists() and not args.overwrite:
        print(f"已存在，跳过：{dst_vitpose}\n（要重算加 --overwrite）")
        return 0

    # 复用已有的检测框和输入视频副本。
    # 只补缺失的——这样 --src 与 --dst 指向同一目录（检测框已由 prep_boxes.py 备好）
    # 也能正常工作，不会把自己复制到自己身上。
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)
    from preprocess import link_or_copy_video      # 脚本自己的目录（src/）就在 sys.path 上
    copied = []
    for rel in ("0_input_video.mp4", "preprocess/bbx.pt"):
        s, d = src / rel, out_dir / rel
        if s.exists() and not d.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".mp4"):
                # 视频走硬链接（一条素材几个输出根就是几份相同的整片视频）。
                # bbx.pt 仍然复制：它只有几百 KB，省不出什么，而 torch.save 是
                # **就地**写盘的——万一哪天有人重存它，共用的 inode 会连着另一个
                # 轨迹目录一起改掉。大文件才值得冒这个 inode 共享的风险。
                how = link_or_copy_video(s, d)
                copied.append(f"{rel}（{how}）")
            else:
                shutil.copy2(s, d)
                copied.append(rel)
    if copied:
        print(f"已复用：{', '.join(copied)}")
    if not (pre_dir / "bbx.pt").exists():
        print(f"缺少检测框 {pre_dir/'bbx.pt'}；先用 prep_boxes.py 或跑一次 run_gvhmr.py 生成。",
              file=sys.stderr)
        return 1

    # ---- rtmlib ----
    try:
        from rtmlib import RTMPose
    except ImportError:
        print("缺少 rtmlib。安装：pip install rtmlib onnxruntime-gpu", file=sys.stderr)
        return 1

    bbx = torch.load(pre_dir / "bbx.pt", map_location="cpu")
    bbx_xyxy = bbx["bbx_xyxy"].numpy().astype(np.float32)

    # **不预读全帧**（原来这里是 `frames.append(f)` 把整段留在内存里）。
    # 那条路的内存是 O(帧数)：4K 横屏素材预处理成 2276×1280，每帧 8.74 MB，
    # 7388 帧 = **64.6 GB**，而这台机器的提交上限是 56.9 GB——它不是在开头炸的，
    # 是读到四千多帧之后才炸，表现成"跑了很久然后失败"（cv2 报 Failed to allocate
    # 8739840 bytes，那个数正好是 2276×1280×3）。竖屏素材每帧只有 2.76 MB，
    # 加上页面文件能硬撑过去，所以这个坑一直没暴露。
    #
    # 帧数从**检测框**拿（同一段视频，框有几条就有几帧），解码挪进推理循环按需做，
    # 内存降到 O(1 帧)。推理本来就是顺序逐帧的，改成流式不改变结果。
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"打不开视频：{video}", file=sys.stderr)
        return 1
    n = len(bbx_xyxy)
    if args.max_frames is not None:
        n = min(n, args.max_frames)
        print(f"⚠ --max-frames {args.max_frames}：这次只算前 {n} 帧，写出的 vitpose.pt 是"
              f"**半截的**。第 4 步 GVHMR 会按两边长度的最小值跑，于是**静默地**只重建"
              f"这么多帧，产物看起来完全正常——除了冒烟测试，别这么用。", file=sys.stderr)
    print(f"视频: {n} 帧（检测框 {len(bbx_xyxy)} 条，逐帧解码）")

    # rtmlib 的 model_input_size 是 (宽, 高)
    in_w, in_h = 288, 384
    print(f"RTMPose-x ONNX: {onnx.name}")
    print(f"输入尺寸: {in_w}x{in_h}   (ViTPose-H 是 192x256，像素数比 {in_w*in_h/(192*256):.2f}x)")
    # to_openpose 必须显式关掉：打开会输出 OpenPose 的 18 点格式，
    # 而 GVHMR 要的是 COCO17（它按 COCO17 定义的 kp_to_joint 映射）。
    pose = RTMPose(onnx_model=str(onnx), model_input_size=(in_w, in_h),
                   to_openpose=False, backend="onnxruntime",
                   device="cpu" if args.device == "dml" else args.device)
    if args.device == "dml":
        # rtmlib 的 device 只认 cpu/cuda/rocm/mps，传别的会在查表时 KeyError。
        # 但它的解码逻辑全部走 self.session，所以直接换掉 session 即可——
        # 既用上 DirectML，又不必手写 SimCC 解码（那正是本项目用 rtmlib 的原因）。
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "DmlExecutionProvider" not in avail:
            print(f"onnxruntime 无 DirectML provider，可用：{avail}。退回 cpu。", file=sys.stderr)
        else:
            pose.session = ort.InferenceSession(
                str(onnx),
                providers=[("DmlExecutionProvider", {}), "CPUExecutionProvider"])
            print(f"已切到 DirectML（实际 provider: {pose.session.get_providers()}）")

    out = np.zeros((n, 17, 3), dtype=np.float32)
    t0 = time.time()
    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                # 视频比检测框短（正常不该发生）：就地产出已经算好的部分，不拿旧帧凑数
                print(f"\n第 {i} 帧读不出来，视频比检测框短；只算前 {i} 帧。", file=sys.stderr)
                out, n = out[:i], i
                break
            if i == 0:
                print(f"      分辨率 {frame.shape[1]}x{frame.shape[0]}")
            kps, scores = pose(frame, bboxes=bbx_xyxy[i : i + 1])
            out[i, :, :2] = kps[0]
            out[i, :, 2] = scores[0]
            del frame                  # 显式放掉这一帧，别等 GC——峰值只该是一帧
            if (i + 1) % 25 == 0 or i == n - 1:
                el = time.time() - t0
                print(f"  {i+1}/{n}  {el:.0f}s  预计剩余 {el/(i+1)*(n-i-1):.0f}s", end="\r", flush=True)
    finally:
        cap.release()

    torch.save(torch.from_numpy(out), dst_vitpose)
    print(f"\n已写入 {dst_vitpose}  ({dst_vitpose.stat().st_size/1024:.0f} KB)")
    print(f"置信度: 均值 {out[:,:,2].mean():.3f}  低于 0.5 占比 {(out[:,:,2]<0.5).mean():.1%}")
    print()
    print("下一步：")
    print(f'  python run_gvhmr.py --video "{video}" -s --output_root "{dst}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
