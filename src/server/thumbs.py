"""素材缩略图：抽一帧当封面。

为什么在服务端抽而不是让浏览器去 seek：列表上一次几十条，浏览器逐条加载视频
再 seek 会同时吃几十个解码器；服务端抽一帧只要一次 ffmpeg（几百毫秒），
而且**抽完就落盘**，第二次打开是纯读文件。

选哪一帧
--------
取片长的 25%（夹在 1~60 秒之间）。开头几秒常是片头/黑场/淡入，固定取第 3 秒
对很多素材会得到一张黑图。取到位置之后再交给 ffmpeg 的 `thumbnail` 滤镜——
它在接下来 25 帧里挑"最有代表性"的一帧（按与邻帧的差异），能躲开一帧的
转场和虚焦。

尺寸
----
统一 **正方形** 128×128（生成尺寸，列表上按 56px 显示），用
`force_original_aspect_ratio=increase` + `crop` 居中裁。

为什么是正方形而不是 16:9：素材绝大多数是**竖屏**，16:9 的居中横带正好把头和脚
都切掉（实测第一版就是这样，只剩一条躯干）。正方形留住中间那块，对"这是哪条视频"
够用，而且横竖屏用的是同一个框，行高不会跳。

落盘位置：`workspace/outputs/<slug>/thumb.jpg`。放这里是因为**删除素材时
cleanup.delete 会整目录 rmtree 掉 outputs/<slug>**，缩略图跟着走，
不会在删掉素材之后留下一张孤儿图（放 web_cache 就会）。
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from . import paths
from batch_run import probe_fps      # paths 已经把 src/ 放进 sys.path

THUMB_W, THUMB_H = 128, 128
SEG_SIZE = 144           # 轨迹段小图（比整帧缩略图大一点，要能看脸认人）   # 生成尺寸（正方形，见上面“尺寸”）
QUALITY = 5                     # ffmpeg -q:v，2 最好 31 最差
AT_FRACTION = 0.25              # 取片长的这个比例处
MIN_AT, MAX_AT = 1.0, 60.0      # 夹取范围（秒）
SCAN_FRAMES = 25                # thumbnail 滤镜在这么多帧里挑一帧

# 帧率缓存：同一条素材要抽几十张轨迹段小图，不该每张 ffprobe 一次。
_FPS: dict[Path, float] = {}


def _fps_of(p: Path) -> float:
    """视频的帧率（带缓存）。

    **必须问正在裁的那条视频**，不能用原片的帧率顶替——见 segment_thumb 里的
    "时刻"那段：框的帧号是在预处理产物上数的，而 60fps 的源会被降到 30fps。
    """
    if p in _FPS:
        return _FPS[p]
    try:
        f = float(probe_fps(p))
    except (OSError, ValueError, subprocess.SubprocessError):
        f = 0.0
    if f > 0:
        _FPS[p] = f
        return f
    return 30.0                      # 探不出来就按 30：宁可差一点也不许抛出去

# 同时最多跑几个 ffmpeg。列表一次会请求几十张缩略图，不设限的话就是几十个
# ffmpeg 进程一起抢 CPU——它们每个只要几百毫秒，但并发几十个会把机器拖垮，
# 反而更慢。串行又太慢（40 张 × 0.4s = 16s），所以给一个小的并发度。
_SEM = threading.Semaphore(4)
# 同一个 slug 别重复抽（列表重渲染会重复发请求）
_LOCK = threading.Lock()
_INFLIGHT: set[str] = set()


def path_of(slug: str) -> Path:
    return paths.OUTPUTS / slug / "thumb.jpg"


def source_video(slug: str) -> Path | None:
    """抽帧用哪条视频。优先预处理产物（更小、seek 更快），没有再用原片。"""
    p = paths.processed_video(slug)
    if p is not None:
        return p
    from . import index
    clip = index.by_slug(slug)
    if clip is None:
        return None
    v = paths.VIDEOS / clip.filename
    return v if v.exists() else None


def ensure(slug: str, duration: float | None = None) -> Path | None:
    """有就返回，没有就抽一张。抽不出来返回 None（调用方 404）。

    **不抛异常**：缩略图失败不该让列表页报错，少一张图而已。
    """
    out = path_of(slug)
    src = source_video(slug)
    if src is None:
        return None
    # 源比缩略图新 = 素材被换过了，重抽
    if out.exists() and out.stat().st_size > 0 and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    with _LOCK:
        if slug in _INFLIGHT:            # 别的请求正在抽同一张，让它去抽
            return out if out.exists() else None
        _INFLIGHT.add(slug)
    try:
        at = _pick_time(duration)
        out.parent.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再改名：抽到一半被打断（或两个请求撞上）不会留下半张图，
        # 而半张 JPEG 会被浏览器当成坏图、而且**下次会被当成已有**不再重抽。
        #
        # 文件名必须是 `thumb.tmp.jpg` 而不是 `thumb.jpg.tmp`——**ffmpeg 靠扩展名
        # 猜封装格式**，`.tmp` 它不认，会直接报"Unable to find a suitable output
        # format"退出。（这个坑真踩了一次：手工跑同样的命令写 /tmp/x.jpg 是好的，
        # 一走模块就 404。）
        tmp = out.with_name(out.stem + ".tmp" + out.suffix)
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{at:.2f}", "-i", str(src),
            "-vf", (f"thumbnail={SCAN_FRAMES},"
                    f"scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=increase,"
                    f"crop={THUMB_W}:{THUMB_H}"),
            "-frames:v", "1", "-q:v", str(QUALITY), str(tmp),
        ]
        with _SEM:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=120)
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(out)
        return out
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        with _LOCK:
            _INFLIGHT.discard(slug)


def _pick_time(duration: float | None) -> float:
    if not duration or duration <= 0:
        return MIN_AT
    return min(max(duration * AT_FRACTION, MIN_AT), MAX_AT)


def clear(slug: str) -> None:
    """删掉一条素材的缩略图（素材被换掉时用得上）。"""
    path_of(slug).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 轨迹段（多人）的小图：**从框里裁**，不是整帧缩略图
# --------------------------------------------------------------------------

def seg_path(slug: str, tid: int) -> Path:
    return paths.OUTPUTS / slug / f"seg_{tid}.jpg"


def segment_thumb(slug: str, tid: int, frame: int, box: list[float],
                  size: int = SEG_SIZE) -> Path | None:
    """把某条轨迹段在某一帧的框裁出来当"这个人长什么样"。

    代表帧由调用方挑（我们挑该段里**框最大**的那帧：人离镜头最近、最清楚）。
    裁切以框中心为中心、放大一点点留白，再等比缩到正方形——直接用框的原始长宽比
    会得到一堆细长的条，列表里没法看。
    """
    src = source_video(slug)
    if src is None:
        return None
    out = seg_path(slug, tid)
    if out.exists() and out.stat().st_size > 0 and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # 边长取框的较长边 ×1.15 留白，保证人整个在画面里
    side = max(x2 - x1, y2 - y1) * 1.15
    # 视频可能比框的分辨率小（框是在预处理产物上算的，两者一致；这里只是防御）
    side = max(16.0, side)
    cw = ch = int(round(side))
    x = int(round(cx - cw / 2))
    y = int(round(cy - ch / 2))
    # **帧号 ÷ 正在裁的这条视频的帧率**，不是原片的。框的帧号是在预处理产物上
    # 数的，而预处理会把 >40fps 的源降到 30fps（batch_run.step_preprocess）——
    # 拿原片的 60 去除，算出来的时刻正好差一倍，裁到的是**另一个时刻的另一个人**
    # （实测 BV1RZY4zPEw2：tid 1 的代表帧 992 该在 33.07s，按原片帧率裁成了 16.53s）。
    at = max(0.0, frame / _fps_of(src))
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp" + out.suffix)
        # 边长用 min(iw,…) 夹回画内：框可能比画面还大（人物特写贴着边），而
        # crop 的**边长**不许超过输入——超了 ffmpeg 直接报 "Invalid too big or
        # non positive size"，小图就静默地没有。**x/y 为负是允许的**（会补边），
        # 所以起点不用夹。让 ffmpeg 自己夹，省一次探测画面尺寸。
        vf = (f"crop=min(iw\\,{cw}):min(ih\\,{ch}):{x}:{y},"
              f"scale={size}:{size}:force_original_aspect_ratio=increase,"
              f"crop={size}:{size}")
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", str(src),
               "-vf", vf, "-frames:v", "1", "-q:v", str(QUALITY), str(tmp)]
        with _SEM:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=120)
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(out)
        return out
    except (OSError, subprocess.SubprocessError):
        return None
