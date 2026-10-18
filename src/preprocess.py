"""视频预处理：降采样 + 可选去抖 + 可选抽帧。

用途
----
1. 4K 竖屏素材直接送姿态管线太重，先降到合理高度。
2. 相机「基本固定但有抖动」时，抖动会被 SLAM/位姿估计错误地吸收进人体运动，
   导致骨盆出现假平移。去抖把相机变成真正静止，之后管线才能安全地用静态相机模式
   （GVHMR 的 `-s`、或我们自己的优化拟合器）。

用法
----
  # 基础：降采样
  python preprocess.py "视频.mp4" -o out

  # 降采样 + 去抖（推荐给你的素材）
  python preprocess.py "视频.mp4" -o out --deshake

  # 顺便抽帧，供 viz 框架使用
  python preprocess.py "视频.mp4" -o out --deshake --frames

  # 调整目标高度和画质
  python preprocess.py "视频.mp4" -o out --height 1920 --crf 16

关于去抖参数
------------
采用 ffmpeg 的 vidstab 两遍法：第一遍估计帧间运动，第二遍反向补偿。
两个参数是刻意设成这样的：

  smoothing=10   低平滑。只削掉高频抖动，保留缓慢的真实位移。
                 调大会连真实运动一起削掉——对运动相机有用，对静止机位有害。
  optzoom=0      关闭自动缩放。
                 这很关键：自动缩放会改变有效焦距，让你估计的相机内参 K 失效，
                 进而污染整个人体系度量结果。宁可留一点黑边也不能让它改焦距。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], desc: str, cwd: str | None = None) -> None:
    """跑一条 ffmpeg/ffprobe 命令，失败时把 stderr 尾部抛出来。

    cwd 用于规避 Windows 路径在 ffmpeg 滤镜参数里的转义问题：
    把工作目录切到临时目录后，滤镜里只需写裸文件名，不必处理 ':' 和 '\\'。
    """
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=cwd)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise RuntimeError(f"{desc} 失败（退出码 {proc.returncode}）:\n{tail}")


def video_readable(path: str | Path, timeout: float = 60.0) -> bool:
    """这个视频文件**能不能解码**。判断依据是 ffprobe 读不读得出视频流。

    **为什么需要这个函数**：MP4 的索引（moov atom）写在文件末尾，编码写到一半
    被打断就会留下一个**大小正常、但谁也读不了**的残件。而"预处理做没做"的判据
    过去只看文件在不在——残件于是被当成已完成产物，整条管线跑到第 2 步才炸，
    报的还是一句离得很远的 `pyav can not handle the given uri`。

    这不是假想的：本项目真的留下了这样一个文件（`BV1zRYQ6AEtE-Lo_1280p_30fps.mp4`，
    2026-10-01 那批跑了一半）。56 个素材里只有它一个，所以平时看不出来。

    退出码非 0、或者什么都输出不出来，就算读不了。文件不存在/零字节也算。
    """
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return False
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def _same_sample(a: Path, b: Path, n: int = 1 << 16) -> bool:
    """头尾各比 n 字节。用于"要删掉目标"之前再确认一道实物证据。

    头 + 尾 128 KB 足以分开两个不一样的视频（容器头、码率、尾包都不同），
    代价是两次 64 KB 读——相对一次 370 MB 的复制可以忽略。**不是**完整性校验，
    完整性校验在 dedupe_input_video.py 里（那是一次性、批量、直接改盘的操作）。
    """
    try:
        with a.open("rb") as fa, b.open("rb") as fb:
            if fa.read(n) != fb.read(n):
                return False
            if a.stat().st_size > 2 * n:
                fa.seek(-n, 2)
                fb.seek(-n, 2)
                return fa.read(n) == fb.read(n)
            return True
    except OSError:
        return False


def link_or_copy_video(src: str | Path, dst: str | Path) -> str:
    """把 src 送到 dst：**优先硬链接**，不行再退回复制。返回做了哪一种。

    为什么要硬链接
    --------------
    GVHMR 的路径约定要求输入视频躺在它自己的输出目录下（`0_input_video.mp4`）。
    而多人轨迹给每条轨迹一个**独立输出根**，于是一条素材有几个输出根就有几份
    逐字节相同的副本——实测盘上 58 份、3.85 GB（`md5sum` 全部相同），
    一个字节的新信息都没有。硬链接让 N 条路径共用同一份数据：路径约定一个字
    不用改，磁盘占用退回 1 份。NTFS / ext4 都支持；跨卷或 FAT32 会抛 OSError，
    这时退回复制，行为跟以前完全一样。

    **必须同时看"新不新"，这不是顺手加的**
    --------------------------------
    GVHMR 的 demo 里那句 `[Copy Video]` 带一个判据 `not exists or 帧数不同`，
    帧数不同时它用 ffmpeg **就地**重写 `0_input_video.mp4`（truncate + 写）。
    如果那份是硬链接，就地重写会连着**源文件**（`workspace/processed/` 里那份）
    一起改掉——所有轨迹目录共用同一个 inode，等于一次写坏一整条素材。
    所以这里保证目标永远等于源（不一致就先 unlink 再重建），让那个判据永远
    走不进"重写"分支。

    返回：
      "kept"    目标已经是同一份数据（同一个 inode），什么都没做
      "linked"  建了硬链接（省下一整份）
      "copied"  链不了，复制了一份（跨卷/不支持的文件系统）
      "relinked"目标是逐字节相同的**真拷贝**（老版本留下的），换成了链接
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        try:
            s, d = src.stat(), dst.stat()
        except OSError:                       # 刚好被删/正在写：当它不存在处理
            s = d = None
        if d is not None and s is not None:
            same_inode = (d.st_ino == s.st_ino and d.st_dev == s.st_dev)
            if same_inode:
                return "kept"
            twin = (d.st_size == s.st_size and int(d.st_mtime) == int(s.st_mtime)
                    and _same_sample(src, dst))
            if not twin:
                # 和源不一样（多半是换过一次预处理）。**不在这里改语义**：
                # 它的内容可能是当时那批产物算出来时用的那份，留着比换掉安全。
                # GVHMR 那边发现帧数不同会自己从源重写一遍。
                return "kept"
            # 大小和 mtime 都一样 = 同一份内容的真拷贝（copy2 保留 mtime），
            # 只是白占一整份空间。换成链接。
            dst.unlink()
            how = "relinked"
        else:
            how = None
    else:
        how = None
    if how is None:
        how = "linked"
    # 调用方不一定把目录建好了（os.link 不会替你建），而且真到写的时候才发现
    # 没目录，报的是"系统找不到指定的路径"这种看不出是缺目录的错。
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        how = "copied"
    return how


def probe(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
        "-show_entries", "format=duration,size",
        "-of", "json", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"无法读取视频信息：\n{(proc.stderr or '').strip()[-500:]}")
    data = json.loads(proc.stdout)
    st = data["streams"][0]
    fmt = data.get("format", {})   # ffprobe 把 duration/size 放在 format 子对象下
    num, den = (st.get("r_frame_rate") or "30/1").split("/")
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": float(num) / float(den or 1),
        "frames": int(st.get("nb_frames") or 0),
        "codec": st.get("codec_name", "?"),
        "duration": float(fmt.get("duration") or 0.0),
        "size": int(fmt.get("size") or 0),
    }


def has_filter(name: str) -> bool:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    return name in (proc.stdout or "")


def pick_scale(info: dict, target_height: int | None) -> tuple[str, int, int]:
    """返回 (scale 滤镜参数, 输出宽, 输出高)。宽度取偶数以兼容 yuv420p。"""
    w, h = info["width"], info["height"]
    if not target_height or target_height >= h:
        out_w, out_h = (w // 2) * 2, (h // 2) * 2
        return f"scale={out_w}:{out_h}", out_w, out_h
    out_h = target_height
    out_w = int(round(w * target_height / h))
    out_w -= out_w % 2                      # H.264 要求偶数宽度
    out_h -= out_h % 2
    return f"scale={out_w}:{out_h}", out_w, out_h


def sizeof(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{num}B"
        num /= 1024.0
    return f"{num:.1f}GB"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    p = argparse.ArgumentParser(description="视频降采样 + 去抖预处理")
    p.add_argument("input", help="输入视频路径")
    p.add_argument("-o", "--outdir", default="processed", help="输出目录（默认 processed）")
    p.add_argument("--name", default=None,
                   help="覆盖输出文件名主干。下游 GVHMR 用 hydra 解析覆盖项，"
                        "**不接受非 ASCII 文件名**，中文名素材必须在这里换成 ASCII")
    p.add_argument("--height", type=int, default=1280,
                   help="目标高度（默认 1280；原始更矮则保持原样）")
    p.add_argument("--crf", type=int, default=18,
                   help="H.264 画质，越小越好越大文件（默认 18，接近无损）")
    p.add_argument("--fps", type=float, default=None,
                   help="输出帧率（默认保持原样）。GVHMR 按 30fps 训练，"
                        "代码里 data/length/30 与静态接触判定的 dt=0.033 都是硬编码——"
                        "60fps 素材必须降到 30，否则模型看到两倍速运动")
    p.add_argument("--deshake", action="store_true", help="启去抖（vidstab 两遍法）")
    p.add_argument("--smoothing", type=int, default=10, help="去抖平滑帧数（默认 10）")
    p.add_argument("--frames", action="store_true", help="额外抽帧导出")
    p.add_argument("--frames-format", choices=["jpg", "png"], default="jpg", help="抽帧格式")
    p.add_argument("--keep-audio", action="store_true", help="保留音轨（默认丢弃，姿态管线用不到）")
    args = p.parse_args(argv)

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"找不到 {tool}，请先安装并加入 PATH。", file=sys.stderr)
            return 1

    src = Path(args.input)
    if not src.exists():
        print(f"输入文件不存在：{src}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.name or src.stem

    info = probe(str(src))
    scale, out_w, out_h = pick_scale(info, args.height)

    # 帧率转换放在缩放之前：先丢掉多余帧再缩放，省一半工作量。
    # fps 滤镜按时间轴重采样，60->30 即每两帧取一帧。
    fps_filter = f"fps={args.fps:g}," if args.fps and abs(args.fps - info["fps"]) > 1e-6 else ""
    chain = f"{fps_filter}{scale}"

    out_fps = args.fps if fps_filter else info["fps"]
    suffix = ("_deshake" if args.deshake else "") + (f"_{out_fps:g}fps" if fps_filter else "")

    print(f"输入   {src.name}")
    print(f"       {info['width']}x{info['height']}  {info['codec']}  "
          f"{info['fps']:.2f}fps  {info['frames']}帧  {info['duration']:.1f}s  {sizeof(info['size'])}")
    print(f"输出   {out_w}x{out_h}  {out_fps:g}fps"
          f"{'  + 去抖' if args.deshake else ''}")

    audio = ["-c:a", "copy"] if args.keep_audio else ["-an"]
    # preset 用 veryfast 而不是 medium：实测（4K60 → 1280p30，30 秒样本）
    #   medium    27.9s   22.0 MB
    #   veryfast  15.6s   20.4 MB   ← 1.8× 快，而且**文件更小**
    #   nvenc p5   9.7s   41.5 MB   ← 2.9× 快，但体积翻倍、且要占 GPU
    # 与 medium 逐帧比对 PSNR 49.6 dB —— 这一步的输出是喂给模型的中间产物，
    # 这个差别看不见。NVENC 更快，但它会和后面的 GPU 步骤抢显卡，而且产物要
    # 长期躺在盘上（每条素材 20~370 MB），所以默认仍然走 CPU 编码。
    enc = ["-c:v", "libx264", "-crf", str(args.crf), "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    src_abs = src.resolve()
    dst = (outdir / f"{stem}_{out_h}p{suffix}.mp4").resolve()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_abs = str(Path(tmp).resolve())
        trf_name = "transforms.trf"          # 裸文件名，配合 cwd 使用

        if args.deshake:
            if not (has_filter("vidstabdetect") and has_filter("vidstabtransform")):
                print("当前 ffmpeg 不含 vidstab 滤镜，请去掉 --deshake 或换一个 ffmpeg 构建。",
                      file=sys.stderr)
                return 1

            # 第一遍：在降采样后的画面上估计帧间运动（省时间，精度足够）
            print("  [1/3] 分析抖动 ...")
            run(["ffmpeg", "-y", "-v", "error", "-i", str(src_abs),
                 "-vf", f"{chain},vidstabdetect=shakiness=8:accuracy=15:result={trf_name}",
                 "-f", "null", "-"], "去抖第一遍（运动估计）", cwd=tmp_abs)

            # 第二遍：补偿。optzoom=0 关掉自动缩放，保住相机内参不变。
            print("  [2/3] 补偿抖动 ...")
            vf = (f"{chain},vidstabtransform=input={trf_name}:smoothing={args.smoothing}"
                  f":optzoom=0:zoom=0:crop=black:interpol=bicubic")
            run(["ffmpeg", "-y", "-v", "error", "-i", str(src_abs),
                 "-vf", vf, *enc, *audio, str(dst)], "去抖第二遍（补偿）", cwd=tmp_abs)
        else:
            print("  [1/1] 转码降采样 ...")
            run(["ffmpeg", "-y", "-v", "error", "-i", str(src_abs),
                 "-vf", chain, *enc, *audio, str(dst)], "转码降采样")

        if args.frames:
            print("  [3/3] 抽帧 ...")
            fdir = (outdir / f"{stem}_{out_h}p{suffix}_frames").resolve()
            fdir.mkdir(parents=True, exist_ok=True)
            q = ["-q:v", "2"] if args.frames_format == "jpg" else []
            run(["ffmpeg", "-y", "-v", "error", "-i", str(dst),
                 "-vsync", "0", *q, str(fdir / f"frame_%06d.{args.frames_format}")], "抽帧")
            n = len(list(fdir.glob(f"*.{args.frames_format}")))
            print(f"        {n} 帧 -> {fdir}")

    out_size = dst.stat().st_size if dst.exists() else 0
    ratio = out_size / info["size"] if info["size"] else 0
    print()
    print(f"完成   {dst}")
    print(f"       {sizeof(out_size)}  (原 {sizeof(info['size'])}，压缩到 {ratio:.0%})")
    print()
    print("下一步：把上面这个 mp4 传到 https://littlefrog-gvhmr.hf.space")
    if args.deshake:
        print("提示：这是去抖版。想对比基线，去掉 --deshake 再跑一次拿原版。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
