"""可行性探针：一条素材里 YOLO/ByteTrack 到底跟出几条轨迹、稳不稳。

**只读**：YOLO 的 track() 不写任何文件，也不碰作业队列。
参数可以是绝对路径，也可以是 workspace/processed 下的文件名。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "shims"))
sys.path.insert(0, str(ROOT / "GVHMR"))
os.chdir(ROOT / "GVHMR")

from hmr4d.utils.preproc.tracker import Tracker          # noqa: E402
from hmr4d.utils.video_io_utils import get_video_lwh     # noqa: E402


def resolve(a: str) -> Path:
    p = Path(a)
    return p if p.is_absolute() else (ROOT / "workspace" / "processed" / a)


t = Tracker()
for arg in sys.argv[1:]:
    path = resolve(arg)
    if not path.exists():
        print(f"{path}: 找不到")
        continue
    L, W, H = get_video_lwh(str(path))
    hist = t.track(str(path))
    id_to_frames, id_to_bbx, order = t.sort_track_length(hist, str(path))
    print(f"\n=== {path.name}   {L} 帧 {W}x{H}   YOLO 跟出 {len(order)} 条轨迹")
    print(f"    {'track':>6} {'出现帧':>7} {'占比':>7} {'面积和':>8}  出现区间")
    for tid, frames in sorted(id_to_frames.items(), key=lambda kv: -len(kv[1]))[:8]:
        f = sorted(frames)
        area = sum((b[2] - b[0]) * (b[3] - b[1]) / W / H for b in id_to_bbx[tid])
        # 有没有断段：断段 = 这人中途消失过（遮挡/出画），换 ID 的特征
        segs = 1
        for i in range(1, len(f)):
            if f[i] != f[i - 1] + 1:
                segs += 1
        span = f"{f[0]}~{f[-1]}"
        print(f"    {tid:>6} {len(f):>7} {len(f) / L:>6.1%} {area:>8.3f}  "
              f"{span}  {'连续' if segs == 1 else f'{segs} 段'}")

    # 逐帧人数分布：两人同时在画面的时段有多长
    per_frame = [len(fr) for fr in hist]
    both = sum(1 for n in per_frame if n >= 2)
    many = sum(1 for n in per_frame if n >= 3)
    print(f"    逐帧人数：同时 >=2 人的帧 {both}（{both / L:.0%}），"
          f">=3 人的帧 {many}（{many / L:.0%}）")
