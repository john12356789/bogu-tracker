"""把各输出根里的 `0_input_video.mp4` 换成指向预处理产物的硬链接。

    python src/dedupe_input_video.py            # 只看，不动（默认）
    python src/dedupe_input_video.py --apply    # 真做

**这是补历史的一遍。** 管线现在已经改成链接优先（见
`preprocess.link_or_copy_video`），但改动之前跑出来的素材，每个输出根里都躺着
一份真的整片拷贝——实测 58 份、3.85 GB，`md5sum` 全部相同，没有一个字节是新的。
这条脚本负责把已有的那些也换掉，不然只有"重跑一遍"才能省下来，而没人会为了
省空间去重跑几十分钟的推理。

安全约束（三道，都过了才动）：

  1. **大小 + mtime 都相同**才算"同一份内容"。`shutil.copy2` 保留 mtime，
     所以对老副本成立；换过一次预处理（内容变了）的一律跳过。
  2. 替换前把两边的 md5 对一遍，不一致就跳过并报出来——不靠推断吃饭。
  3. 替换是**原子**的：先在旁边建链接再 `os.replace` 顶上去，中途没有"文件
     不存在"的窗口。（同一个卷上 rename 是原子的。）

内容和 inode 都换不回来的是"删了源"，所以源不存在时一律跳过、不猜。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from server import paths  # noqa: E402

NAME = "0_input_video.mp4"


def md5(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="把 0_input_video.mp4 换成硬链接")
    ap.add_argument("--apply", action="store_true",
                    help="真的替换（不加就只报告）")
    ap.add_argument("--no-hash", action="store_true",
                    help="跳过 md5 校验（快，但只靠大小+mtime 判断）")
    args = ap.parse_args()

    cands = sorted(paths.OUTPUTS.glob(f"*/*/*/{NAME}")) + sorted(paths.OUTPUTS.glob(f"*/*/{NAME}"))
    if not cands:
        print("没找到任何输入视频副本")
        return 0

    src_md5: dict[Path, str | None] = {}
    done: list[tuple[Path, int]] = []
    skipped: list[tuple[Path, str]] = []

    for dst in cands:
        src = paths.PROCESSED / f"{dst.parent.name}.mp4"
        try:
            s, d = src.stat(), dst.stat()
        except OSError as exc:
            skipped.append((dst, f"读不到：{exc.__class__.__name__}"))
            continue
        if not src.exists():
            skipped.append((dst, f"源不在：{src.name}"))
            continue
        if (s.st_dev, s.st_ino) == (d.st_dev, d.st_ino):
            skipped.append((dst, "已经是链接"))
            continue
        if d.st_size != s.st_size:
            skipped.append((dst, f"和源大小不同（{d.st_size} vs {s.st_size} 字节）"))
            continue
        # mtime 只在不做内容校验时才当判据。实测有一条素材的预处理产物被**重做过
        # 一次、结果逐字节相同**（md5 一模一样，只是 mtime 新了）——那种也不该
        # 白占一整份。大小相同才往下走，然后内容说了算。
        if args.no_hash and int(d.st_mtime) != int(s.st_mtime):
            skipped.append((dst, "mtime 不同（没开内容校验，保守跳过）"))
            continue
        if not args.no_hash:
            if src not in src_md5:
                src_md5[src] = md5(src)
            if md5(dst) != src_md5[src]:
                skipped.append((dst, "**内容不一样**（大小和 mtime 撞上了）"))
                continue
        if not args.apply:
            done.append((dst, d.st_size))
            continue
        # 原子替换：先建链接到临时名，再 rename 顶上去
        tmp = dst.with_suffix(".dedupe-tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
            os.replace(tmp, dst)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            skipped.append((dst, f"链接失败：{exc}"))
            continue
        now = dst.stat()
        if (now.st_dev, now.st_ino) != (s.st_dev, s.st_ino):
            skipped.append((dst, "替换后 inode 不对（已放弃）"))
            continue
        done.append((dst, d.st_size))

    total = sum(n for _, n in done)
    for p, n in done:
        print(f"  {'替换' if args.apply else '可省'} {n/1e6:8.1f} MB  {p.relative_to(paths.OUTPUTS)}")
    for p, why in skipped:
        print(f"  跳过            {p.relative_to(paths.OUTPUTS)}  —— {why}")
    print()
    verb = "已替换，释放" if args.apply else "可释放（加 --apply 执行）"
    print(f"{len(done)} 份{verb} {total/1e9:.2f} GB；{len(skipped)} 份跳过，原样不动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
