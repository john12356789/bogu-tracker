"""只跑检测框和 HMR2 特征，跳过 ViTPose。

为什么需要它
------------
GVHMR 的 `run_preprocess` 把「YOLO 检测 -> ViTPose 关键点 -> HMR2 特征」写死成一条链，
没有开关。当我们用 RTMPose 替换 2D 关键点时，ViTPose 那一步的产物会被直接丢弃——
纯属白跑（每个视频 1~2 分钟）。

这个脚本只做两件事：
  Tracker   -> bbx.pt           检测框（YOLO）
  Extractor -> vit_features.pt  HMR2 特征（只依赖图像和检测框）

剩下的 vitpose.pt 交给 run_rtmpose.py 填。三份都齐了之后，
run_gvhmr.py 的 run_preprocess 会因为产物已存在而全部跳过，直接进推理。

用法
----
    python prep_boxes.py --video <视频> --output-root <绝对路径>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GVHMR_ROOT = ROOT / "GVHMR"
SHIMS = HERE / "shims"


def _extract_streaming(extractor, video_path: str, bbx_xys, img_ds: float,
                       batch_size: int = 16, chunk: int = 256, half: bool = True):
    """分块 + 分批提取 HMR2 特征，绕开上游两个**随视频长度线性增长**的占用。

    上游 hmr4d/utils/preproc/vitfeat_extractor.py 的 get_batch + extract_video_features：
      1. `read_video_np` 把**整段**视频读进内存（scale=0.5 时 4405 帧 ≈ 3.0GB）；
      2. 高斯模糊那步再复制一份（再 3.0GB）；
      3. 裁切后堆成 (F,3,256,256) float32（再 3.5GB）；
      4. 最后 `imgs.cuda()` 把**整段**搬上显存（3.5GB）。
    4405 帧时内存峰值约 9.5GB、显存 3.5GB——16GB 内存 / 6GB 显存上会打满，
    既拖慢（实测长片 290ms/帧 vs 短片 68ms/帧）也有 OOM 风险。

    这里按 chunk 帧读、按 batch 进显存，两处占用都与视频长度解耦。
    **数值与上游逐元素一致**：同样的模糊参数、同样的裁剪、同样的批顺序。

    `half=True`（默认）让 ViT-Huge 的前向走 fp16（autocast）。**实测过**（BV1RZY4zPEw2，
    2596 帧，见 src/_fp16_ab.py）：

        特征耗时   81.5 -> 43.0 ms/帧（1.90x）
        特征数值   逐元素相对差 1.7e-3
        -> 关节    99.82% 在 1 mm 内，最大 15.1 mm（第 1673 帧右脚，单帧单关节）
        -> 最终脚本 **六条轴最大差 1/100 点，没有任何一帧超过 1 个点**
                    （funscript 的整数分辨率就是 1 个点，也就是差在量化地板上）

    对照组的 fp32 重算与现网产物**逐元素相同**（0.000 mm），所以上面的差确实是
    fp16 造成的，不是跑批噪声。存盘仍是 float32，文件格式不变。

    不想要这点数值差就用 `--fp32` 关掉（慢一倍，换回逐元素一致的旧行为）。
    """
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    from hmr4d.utils.video_io_utils import read_video_np
    from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD

    F = int(bbx_xys.shape[0])
    dst_size = 256
    gt_center, gt_size = bbx_xys[:, :2], bbx_xys[:, 2]
    feats = []
    bar = tqdm(total=F, desc="HMR2 Feature", leave=getattr(extractor, "tqdm_leave", True))
    for a in range(0, F, chunk):
        b = min(a + chunk, F)
        raw = read_video_np(video_path, start_frame=a, end_frame=b, scale=img_ds)
        # 复刻 get_batch 的自适应模糊：降采样倍率越大、模糊越强
        dsf = ((gt_size[a:b] * img_ds) / dst_size / 2.0).numpy()
        raw = np.stack([cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                        for v, d in zip(raw, dsf)])
        imgs = [crop_and_resize(raw[i], gt_center[a + i] * img_ds,
                                gt_size[a + i] * img_ds, dst_size, enlarge_ratio=1.0)[0]
                for i in range(b - a)]
        imgs = torch.from_numpy(np.stack(imgs))                 # (n,256,256,3) uint8
        imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)
        for j in range(0, imgs.shape[0], batch_size):
            batch = imgs[j:j + batch_size].cuda()
            with torch.no_grad():
                if half:
                    with torch.autocast("cuda", dtype=torch.float16):
                        f = extractor.extractor({"img": batch})
                else:
                    f = extractor.extractor({"img": batch})
            # **一律转回 float32 再存**：autocast 只影响算得快不快，
            # 不影响产物格式——下游和存档都不用知道这件事。
            f = f.detach().float().cpu()
            # fp16 的溢出是**静默**的（变 inf/nan 而不报错），而 NaN 特征会让
            # 整条重建变成垃圾却看着像跑成功了。宁可在这里停住并说清楚。
            if half and not torch.isfinite(f).all():
                raise RuntimeError(
                    f"HMR2 特征出现 inf/nan（fp16 溢出），第 {a + j}~{a + j + f.shape[0]} 帧。"
                    f"加 --fp32 重跑这一步。")
            feats.append(f)
        bar.update(b - a)
        del raw, imgs                                            # 及时释放，别等 GC
    bar.close()
    return torch.cat(feats, dim=0).clone()


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="只跑检测框 + HMR2 特征")
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-root", required=True, help="绝对路径")
    ap.add_argument("--fp32", action="store_true",
                    help="特征提取不用 fp16（慢约 1.9x，换回逐元素一致的旧数值）")
    ap.add_argument("--track", type=int, default=None, metavar="ID",
                   help="只保留指定 track 的框（多人素材用）。不给则用"
                        "「累计占画面最大」的那条（= 本项目一直以来的行为）。"
                        "全部轨迹的原始信息总是会另存一份 bbx_all.json。")
    ap.add_argument("--merge-tids", default=None, metavar="LIST",
                   help="把这几条轨迹段合并成**一条**（逗号分隔的 tid，例如 "
                        "2,8,15）。同一个人被 ByteTrack 切成好几段时用，分组的来历"
                        "见 tracklet_merge.py。合并后的编号取组内最小的 tid。"
                        "和 --track 互斥（两个都给时以这里为准）。")
    ap.add_argument("--tracks-from", default=None, metavar="JSON",
                   help="复用另一份 bbx_all.json 里的轨迹段，**不重跑 YOLO**。"
                        "逐条跑多人素材时用：跟踪结果对所有 track 是同一份，"
                        "每条都重跟一遍纯属白等（实测 5192 帧要 2 分钟）")
    ap.add_argument("--min-coverage", type=float, default=0.03, metavar="F",
                   help="一条轨迹至少要出现在 F 比例的帧里才算「有效」，默认 0.03。"
                        "实测 5192 帧的素材上 YOLO 会跟出 6 条，其中两条只占 0.3~0.4%%"
                        "（路过的观众/误检）——它们进不了 bbx_all.json，也不会被跑管线")
    ap.add_argument("--img-ds", type=float, default=0.5,
                    help="HMR2 特征提取前的帧降采样倍率（上游默认 0.5，即先把帧降到半分辨率）。"
                         "1.0 = 用全分辨率帧。只影响 vit_features.pt，不动检测框和 2D 关键点。"
                         "用它做实验时要输出到**新目录**——产物名固定是 vit_features.pt，"
                         "同目录换值会被「已存在就跳过」的逻辑误判为已完成")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))              # 本目录：preprocess.video_readable
    sys.path.insert(0, str(SHIMS))
    sys.path.insert(0, str(GVHMR_ROOT))
    os.chdir(GVHMR_ROOT)                       # demo.py 里有相对路径
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from preprocess import link_or_copy_video, video_readable

    from hmr4d.utils.pylogger import Log
    from hmr4d.utils.preproc import Tracker, Extractor
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from hmr4d.utils.video_io_utils import get_video_lwh
    from hmr4d.utils.seq_utils import (
        frame_id_to_mask, get_frame_id_list_from_mask, linear_interpolate_frame_ids,
        rearrange_by_mask,
    )

    video = Path(args.video).resolve()
    if not video.exists():
        print(f"找不到视频：{video}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_root).resolve() / video.stem
    pre_dir = out_dir / "preprocess"
    pre_dir.mkdir(parents=True, exist_ok=True)

    # 输入视频副本（GVHMR 的路径约定要求它在 output_dir 下）
    #
    # **要验能不能读，而且不能只看在不在**（两个坑都踩过）：
    #   · 源是残件时 copy2 照样成功——拷的是字节，不是可解码的视频；
    #   · 上一次留下的副本可能正是那种残件，而它一旦存在就不会被重拷，
    #     于是"把预处理重跑好了"也修不好这里。
    #
    # 落地方式用**硬链接**（见 preprocess.link_or_copy_video）：一条素材有几个
    # 输出根就有几份逐字节相同的副本，实测盘上 58 份、3.85 GB 全是重复的。
    # 链不了（跨卷/FAT32）会自动退回复制，行为和以前一样。
    dst_video = out_dir / "0_input_video.mp4"
    if dst_video.exists() and not video_readable(dst_video):
        Log.info(f"[io] 输入副本读不了，丢掉重来：{dst_video.name}")
        dst_video.unlink(missing_ok=True)
    how = link_or_copy_video(video, dst_video)
    if how != "kept":
        Log.info(f"[io] 输入视频 {how}：{dst_video.name}")
    if not video_readable(dst_video):
        print(f"输入视频读不了：{video}\n"
              "多半是预处理产物是残件——编码写到一半被打断，MP4 的索引（moov）"
              "没写出来，文件大小正常但谁也解不了。\n"
              "处理办法：删掉 workspace/processed/ 下这条素材的产物，重跑预处理。",
              file=sys.stderr)
        return 1

    bbx_path = pre_dir / "bbx.pt"
    feat_path = pre_dir / "vit_features.pt"

    # ---- 1) 检测框 ----
    #
    # **多人**：GVHMR 的 Tracker 本来就跟踪所有人（YOLO+ByteTrack，每帧带 track id），
    # 只是 get_one_track() 把除第一条之外的**全扔了**。这里改成：全部轨迹段另存一份
    # bbx_all.json（界面选人 + 逐条跑管线都靠它），而 bbx.pt 仍然只放**一条**——
    # 默认是"累计占画面最大"的那条，和本项目一直以来的行为逐字节一致。
    #
    # 实测（5192 帧的双人素材）YOLO 跟出 6 条：两个人各被切成 9~12 段（互相遮挡时
    # 反复丢失重捕），外加 2 条 0.3% 的噪声。所以**一个人 ≠ 一条 track**，
    # 一个人是一组 track 段——这个区分决定了界面的形态。
    all_path = out_dir / "bbx_all.json"

    # 合并规格（这一根里的框是哪几段并出来的）。**它变了，产物就得作废**：
    # 框换了，HMR2 特征就是在别的裁剪上算的了，而下一步是"文件在就跳过"——
    # 不作废就会把旧特征当成新的喂给 GVHMR，出来的动作对不上框，还很难查。
    want_merge = _parse_tids(args.merge_tids)
    if bbx_path.exists():
        old = _merge_spec_of(bbx_path)
        if old != want_merge:
            Log.info(f"[bbx] 合并规格变了（{old} -> {want_merge}），作废这一根的框和特征")
            bbx_path.unlink(missing_ok=True)
            feat_path.unlink(missing_ok=True)
    if bbx_path.exists():
        Log.info(f"[bbx] 已存在，跳过：{bbx_path}")
    elif args.tracks_from and Path(args.tracks_from).exists():
        # 复用现成的轨迹段：跟踪结果对所有 track 是同一份，逐条跑时没必要重跟
        info = json.loads(Path(args.tracks_from).read_text(encoding="utf-8"))
        L, W, H = info["n_frames"], info["width"], info["height"]
        valid = info["segments"]
        segs = valid
        Log.info(f"[bbx] 复用轨迹清单：{Path(args.tracks_from).name}（{len(valid)} 条有效）")
    else:
        Log.info("[bbx] YOLO 检测 + 跟踪")
        tracker = Tracker()
        track_history = tracker.track(str(dst_video))
        id_to_frames, id_to_bbx, order = tracker.sort_track_length(track_history, str(dst_video))
        del tracker
        L, W, H = get_video_lwh(str(dst_video))

        # 全部轨迹落盘（含被滤掉的噪声，只是标出来），界面和 batch_run 都读它
        segs = []
        for tid in order:
            paired = sorted(zip((int(f) for f in id_to_frames[tid]), id_to_bbx[tid]),
                            key=lambda fb: fb[0])
            frames = [f for f, _ in paired]
            boxes = [[round(float(v), 1) for v in b] for _, b in paired]
            cover = len(frames) / max(1, L)
            # 出现区间（连续段）。界面画"人物带"、拼接时判接缝都用它
            runs, start = [], frames[0]
            for i in range(1, len(frames)):
                if frames[i] != frames[i - 1] + 1:
                    runs.append([start, frames[i - 1]])
                    start = frames[i]
            runs.append([start, frames[-1]])
            segs.append({
                "tid": int(tid), "coverage": round(cover, 4),
                "frames": frames, "boxes": boxes, "runs": runs,
                "area_sum": round(sum((b[2] - b[0]) * (b[3] - b[1]) / W / H for b in boxes), 4),
                "_valid": cover >= args.min_coverage,
            })
        valid = [x for x in segs if x["_valid"]]
        all_path.write_text(json.dumps({
            "version": 1, "video": dst_video.name, "n_frames": L, "width": W, "height": H,
            "min_coverage": args.min_coverage,
            # 有效轨迹按"累计面积"排，界面和跑管线的顺序都用这个
            "segments": [{k: v for k, v in x.items() if k != "_valid"} for x in valid],
            # 被滤掉的噪声轨**也带上 frames/boxes**：归并要把它们算进去（一条
            # 2% 的碎片常常是同一个人被遮挡那几秒里唯一被检出的痕迹，并进来
            # 覆盖率就上去了）。以前这里只留 {tid, coverage}，够界面报个数字，
            # 不够归并用来判断——少的那点体积是几百帧的框，可以忽略。
            "dropped": [{k: v for k, v in x.items() if k != "_valid"} for x in segs
                        if not x["_valid"]],
        }, ensure_ascii=False), encoding="utf-8")
        Log.info(f"[bbx] 全部轨迹 -> {all_path.name}："
                 f"{len(valid)} 条有效（另有 {len(segs) - len(valid)} 条占比 "
                 f"< {args.min_coverage:.0%} 的噪声轨已丢弃）")


    # 只有真需要写盘时才走。**两条分支都要到这里**——复用清单那条也要写
    # bbx.pt，否则逐条跑时目标目录里没有输入，rtmpose 直接找不到文件。
    if not bbx_path.exists():
        # 挑一条当 bbx.pt：**两条分支都要走**（复用清单那条也要写盘，
        # 否则逐条跑时目标目录里没有 bbx.pt，下游 rtmpose 直接找不到输入）。
        # 选哪一条当 bbx.pt
        pick = None
        if want_merge:
            from tracklet_merge import union_pick
            pick = union_pick(segs, want_merge)      # 用**全部** segs，不只 valid
            if pick is None:
                print(f"要合并的轨迹 {want_merge} 一条都不在 "
                      f"{[x['tid'] for x in segs]} 里", file=sys.stderr)
                return 1
            missing = sorted(set(want_merge) - set(pick["tids"]))
            if missing:
                Log.info(f"[bbx] 合并清单里的 {missing} 不存在，已忽略")
            Log.info(f"[bbx] 合并 {len(pick['tids'])} 段 -> track {pick['tid']}"
                     f"（{pick['tid']} = {'+'.join(str(t) for t in pick['tids'])}）")
        elif args.track is None:
            pick = valid[0] if valid else (segs[0] if segs else None)
        else:
            match = [x for x in segs if x["tid"] == args.track]
            if not match:
                print(f"找不到 track {args.track}；这条素材跟出的是 "
                      f"{[x['tid'] for x in segs]}", file=sys.stderr)
                return 1
            pick = match[0]
        if pick is None:
            print("这条素材一帧都没检出人。", file=sys.stderr)
            return 1
        if want_merge:
            pass                                  # 上面已经报过合并了哪几段
        elif args.track is not None:
            Log.info(f"[bbx] 指定 track {pick['tid']}（占 {pick['coverage']:.0%}，"
                     f"{len(pick['runs'])} 段）")
        elif len(valid) > 1:
            Log.info(f"[bbx] 多人素材：bbx.pt 取 {pick['tid']}（占比 "
                     f"{pick['coverage']:.0%}），其余各跑一遍见 batch_run")

        # 稠密 (F,4)：缺的帧插值补上。和 get_one_track 里那套完全一样，
        # 只是现在可以指定是哪一条轨迹。
        fid = torch.tensor(pick["frames"])
        mask = frame_id_to_mask(fid, L)
        dense = rearrange_by_mask(torch.tensor(pick["boxes"], dtype=torch.float32), mask)
        dense = linear_interpolate_frame_ids(dense, get_frame_id_list_from_mask(~mask))
        assert (dense.sum(1) != 0).all(), "插值后仍有空帧"
        bbx_xyxy = dense.float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        # `merge_tids` 是**这一根里产物属于哪几段的凭证**：下次进来先比它，
        # 对不上就把框和特征一起作废（见上面）。老产物没有这个键 = None，
        # 和"没有合并"一致，所以已有素材不会被这次改动逼着重跑。
        # 记的是**真正并进去的那几段**（pick["tids"]），不是命令行要的那一份：
        # 清单里有点不存在的 tid 时两者会不一样，记命令行那份会让下次一进来
        # 就判成"规格变了"，白作废一次产物。
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys,
                    "track_id": pick["tid"],
                    "merge_tids": pick["tids"] if want_merge else None}, bbx_path)
        Log.info(f"[bbx] 已保存：{bbx_path}（track {pick['tid']}）")

    bbx_xys = torch.load(bbx_path, map_location="cpu")["bbx_xys"]

    # ---- 2) HMR2 特征 ----
    if feat_path.exists():
        Log.info(f"[feat] 已存在，跳过：{feat_path}")
    else:
        Log.info(f"[feat] HMR2 特征提取（img_ds={args.img_ds:g}，按批搬显存）")
        extractor = Extractor()
        feats = _extract_streaming(extractor, str(dst_video), bbx_xys, args.img_ds,
                                   half=not args.fp32)
        torch.save(feats, feat_path)
        del extractor
        Log.info(f"[feat] 已保存：{feat_path}")

    length, w, h = get_video_lwh(str(dst_video))
    print()
    print(f"完成：{out_dir}")
    print(f"  视频 {length} 帧 {w}x{h}")
    print(f"  bbx.pt          {'OK' if bbx_path.exists() else '缺失'}")
    print(f"  vit_features.pt {'OK' if feat_path.exists() else '缺失'}")
    print()
    print("下一步：用 run_rtmpose.py 生成 vitpose.pt，然后跑 run_gvhmr.py。")
    return 0


def _parse_tids(spec: str | None) -> list[int] | None:
    """命令行 "2,8,15" -> [2, 8, 15]。空值/坏值给 None（= 不合并）。

    去重 + 排序：合并清单的**内容**才是判据（见 _merge_spec_of），顺序和重复
    不该让产物被无谓地作废一次。
    """
    if not spec:
        return None
    try:
        tids = sorted({int(x) for x in str(spec).replace("，", ",").split(",") if x.strip()})
    except ValueError:
        return None
    return tids or None


def _merge_spec_of(bbx_path) -> list[int] | None:
    """读一根里 bbx.pt 记的合并规格。读不出来（老产物 / 残件）就当 None——
    "读不出"和"没合并"等价，都是不许因此把已有产物判成废件。"""
    import torch
    try:
        return torch.load(bbx_path, map_location="cpu").get("merge_tids")
    except Exception:                                   # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())
