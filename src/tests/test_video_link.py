"""输入视频用硬链接铺开：省空间，但不许改掉内容。

    python src/tests/test_video_link.py

背景：GVHMR 的路径约定要求输入视频躺在自己的输出目录下，而多人轨迹给每条轨迹
一个独立输出根。于是一条素材有几个输出根就有几份**逐字节相同**的整片视频——
实测盘上 58 份、3.85 GB，`md5sum` 全部相同。硬链接把 N 条路径并到一份数据上。

这个函数有个**危险动作**：发现目标与源是同一份内容的真拷贝时，它会
`unlink` 目标再建链接。所以这里盯住三件事：

  1. 该链的链上（同一个 inode，内容一模一样，读得出来）
  2. **不该动的绝不删**——和源不一样的（内容不同/大小不同）必须原样留着。
     这条是重点：宁可多占一份空间，也不能把一个已经算完的素材的输入视频换掉。
  3. 链不了的时候要退回复制，而不是报错——跨卷/FAT32 上也要能用。
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import link_or_copy_video  # noqa: E402
from server import cleanup  # noqa: E402

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def write(p: Path, data: bytes, mtime: float | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def same_inode(a: Path, b: Path) -> bool:
    sa, sb = a.stat(), b.stat()
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="linktest_"))
    try:
        return _run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(tmp: Path) -> int:
    # 造一个"够大"的源：_same_sample 要能读出头尾
    blob = bytes(range(256)) * 1024          # 256 KB
    src = write(tmp / "processed" / "V_1280p_30fps.mp4", blob)

    print("\n[1] 目标不存在 → 建链接")
    dst1 = tmp / "out" / "a" / "0_input_video.mp4"
    how = link_or_copy_video(src, dst1)
    check(how == "linked", f"返回 linked（实际 {how}）")
    check(dst1.read_bytes() == blob, "内容一致")
    check(same_inode(src, dst1), "和源是同一个 inode")
    check(dst1.stat().st_nlink == 2, f"硬链接数 2（实际 {dst1.stat().st_nlink}）")

    print("\n[2] 已经是链接 → 什么都不做")
    how = link_or_copy_video(src, dst1)
    check(how == "kept", f"返回 kept（实际 {how}）")
    check(same_inode(src, dst1), "还是同一个 inode")

    print("\n[3] 老版本留下的真拷贝（copy2 保留 mtime）→ 换成链接")
    dst2 = tmp / "out" / "b" / "0_input_video.mp4"
    dst2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst2)                        # copy2 会保留 mtime
    check(not same_inode(src, dst2), "先确认它是独立的一份")
    check(dst2.stat().st_size == src.stat().st_size, "大小相同（所以会被认出来）")
    how = link_or_copy_video(src, dst2)
    check(how == "relinked", f"返回 relinked（实际 {how}）")
    check(same_inode(src, dst2), "换成了链接")
    check(dst2.read_bytes() == blob, "内容一字没变")

    print("\n[4] 和源不一样的一律不动（这是最要紧的一条）")
    dst3 = tmp / "out" / "c" / "0_input_video.mp4"
    other = b"another video entirely" * 4096       # 大小不同
    write(dst3, other)
    before = dst3.read_bytes()
    how = link_or_copy_video(src, dst3)
    check(how == "kept", f"返回 kept（实际 {how}）")
    check(dst3.read_bytes() == before, "内容原封不动")
    check(not same_inode(src, dst3), "没有被换成链接")

    dst4 = tmp / "out" / "d" / "0_input_video.mp4"
    # 大小相同、mtime 也相同，但内容不同——只靠大小+mtime 会被误判成"双胞胎"
    same_size = bytearray(blob)
    same_size[0] ^= 0xFF                           # 改头
    same_size[-1] ^= 0xFF                          # 改尾
    write(dst4, bytes(same_size), mtime=src.stat().st_mtime)
    check(dst4.stat().st_size == src.stat().st_size, "大小和源一样")
    check(int(dst4.stat().st_mtime) == int(src.stat().st_mtime), "mtime 也和源一样")
    how = link_or_copy_video(src, dst4)
    check(how == "kept", f"内容不同 → 仍然不动（实际 {how}）")
    check(dst4.read_bytes() == bytes(same_size), "内容原封不动")

    print("\n[5] 链不了就退回复制，不报错")
    real_link = os.link

    def boom(*a, **kw):
        raise OSError(1, "cross-device link")

    os.link = boom
    try:
        dst5 = tmp / "out" / "e" / "0_input_video.mp4"
        how = link_or_copy_video(src, dst5)
    finally:
        os.link = real_link
    check(how == "copied", f"返回 copied（实际 {how}）")
    check(dst5.read_bytes() == blob, "内容一致")
    check(not same_inode(src, dst5), "是一份独立拷贝")

    print("\n[6] 删除弹窗里的体积要按 inode 去重（否则报得比实际能腾出的大）")
    d = tmp / "usage"
    write(d / "x" / "0_input_video.mp4", blob)
    (d / "y").mkdir(parents=True, exist_ok=True)
    os.link(d / "x" / "0_input_video.mp4", d / "y" / "0_input_video.mp4")
    total, files = cleanup._tree_size(d)
    check(total == len(blob), f"同一份数据只算一次：{total} == {len(blob)}")
    check(files == 2, f"文件数仍按路径算：{files} == 2")

    print()
    if FAILS:
        print(f"✗ {len(FAILS)} 项没过：")
        for f in FAILS:
            print("   -", f)
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
