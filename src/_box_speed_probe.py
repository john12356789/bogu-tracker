"""量一下「检测框 + HMR2 特征」这一步里，跟踪和特征各占多少。

    E:\\anaconda3\\envs\\gvhmr\\python.exe src/_box_speed_probe.py <样本视频>

样本视频用 10 秒（300 帧）的 1280p 片段，然后把每帧耗时外推到大素材上。
下划线开头：这不是测试，是一次性探针。
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GVHMR = ROOT / "GVHMR"
# shim 必须在 GVHMR 之前进 sys.path（上游 import pytorch3d，而 Windows 没有 wheel）
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "shims"))
sys.path.insert(0, str(GVHMR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.chdir(GVHMR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hmr4d.utils.preproc import Extractor, Tracker  # noqa: E402
from hmr4d.utils.seq_utils import (  # noqa: E402
    frame_id_to_mask, get_frame_id_list_from_mask, linear_interpolate_frame_ids,
    rearrange_by_mask,
)
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy  # noqa: E402
from prep_boxes import _extract_streaming  # noqa: E402

video = sys.argv[1]

t = time.time()
tracker = Tracker()
hist = tracker.track(video)
t_track = time.time() - t

id_to_frames, id_to_bbx, order = tracker.sort_track_length(hist, video)
tid = order[0]
paired = sorted(zip((int(f) for f in id_to_frames[tid]), id_to_bbx[tid]), key=lambda fb: fb[0])
F = len(paired)
print(f"[YOLO v8x + ByteTrack] {t_track:.1f}s / {F} 帧 = {t_track / F * 1000:.1f} ms/帧", flush=True)

# 稠密 bbx（和 prep_boxes 里一样的做法：缺的帧线性插值）
all_frames = sorted(int(f) for f in hist[list(hist)[0]]) if isinstance(hist, dict) else []
L = max(int(f) for f in id_to_frames[tid]) + 1
fid = torch.tensor([f for f, _ in paired])
mask = frame_id_to_mask(fid, L)
dense = rearrange_by_mask(torch.tensor([b for _, b in paired], dtype=torch.float32), mask)
dense = linear_interpolate_frame_ids(dense, get_frame_id_list_from_mask(~mask))
bbx_xys = get_bbx_xys_from_xyxy(dense.float(), base_enlarge=1.2).float()
print(f"稠密 bbx: {tuple(bbx_xys.shape)}", flush=True)

t = time.time()
extractor = Extractor()
feats = _extract_streaming(extractor, video, bbx_xys, 0.5)
t_feat = time.time() - t
print(f"[HMR2 特征] {t_feat:.1f}s / {L} 帧 = {t_feat / L * 1000:.1f} ms/帧", flush=True)
print(f"特征形状 {tuple(feats.shape)}", flush=True)

for n, what in ((6727, "BV1A（4K60 / 6727 帧）"), (2596, "BV1RZY4zPEw2（2596 帧）")):
    print(f"外推到 {what}: 跟踪 {t_track / F * n:.0f}s + 特征 {t_feat / L * n:.0f}s "
          f"= {t_track / F * n + t_feat / L * n:.0f}s", flush=True)
