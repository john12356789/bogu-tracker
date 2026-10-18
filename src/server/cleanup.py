"""删除一条素材的产物（可选连原片一起）。

**为什么单开一个模块**：删东西这件事必须只有一处实现。素材的产物散在四个地方
（processed / outputs / funscripts / web_cache 的草稿），少删一处就会留下"半条
素材"——界面上它消失了，磁盘上还压着几个 GB；多删一处更糟。这里把"一条素材
到底由哪些文件构成"写成一份清单，`usage()` 和 `delete()` 共用它，界面上显示的
体积和实际删的东西必然一致。

删除是**不可逆**的，所以：
  · 原片默认不删（`with_video=False`），要删得显式勾
  · 名字先过白名单（slug 会进 URL，`../../` 这种拼进来就能删到别处）
  · 有正在跑或排队的作业时拒绝（否则作业会往删掉的目录里写，或者把它再建出来）
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from . import convert, index, motion, paths

# 素材短名由 ascii_slug() 生成，只可能是这些字符。**必须校验**：slug 来自 URL，
# 拼上 `..` 之后 `OUTPUTS / slug` 就跑出工作区了。
SAFE_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")


def _tree_size(d: Path) -> tuple[int, int]:
    """目录总字节数 + 文件数。不存在返回 (0, 0)。

    **同一个 inode 只算一次**：输入视频是按硬链接铺开的（见
    preprocess.link_or_copy_video），一条素材几个输出根就指向同一份数据。
    按路径累加会把同一份视频算成好几份，而删除时只释放一份——报出去的数字
    比实际能腾出的空间大，用户按它做决定就会失望。

    文件数（第二个返回值）仍按路径算：那是"有多少个文件要删"，不是体积。
    """
    if not d.is_dir():
        return 0, 0
    total = files = 0
    seen: set[tuple[int, int]] = set()
    for p in d.rglob("*"):
        try:
            if p.is_file():
                st = p.stat()
                key = (st.st_dev, st.st_ino)
                if key not in seen:
                    seen.add(key)
                    total += st.st_size
                files += 1
        except OSError:          # 刚好被删/正在写：算不准就不算它
            continue
    return total, files


def _glob_size(pat: str) -> tuple[int, int]:
    total = files = 0
    for p in paths.PROCESSED.glob(pat):
        try:
            total += p.stat().st_size
            files += 1
        except OSError:
            continue
    return total, files


def usage(slug: str) -> dict:
    """这条素材占了哪些东西、各多少字节。删除前给用户看的清单。"""
    if not SAFE_SLUG.match(slug):
        return {}

    out_bytes, out_files = _tree_size(paths.OUTPUTS / slug)
    proc_bytes, proc_files = _glob_size(f"{slug}_*p*.mp4")
    fs_bytes, fs_files = _tree_size(paths.FUNSCRIPTS / slug)
    draft_bytes, draft_files = _tree_size(convert.draft_dir(slug))

    # 原片：按规则反算（见 paths.videos_for）。**不能前缀 glob**：那样
    # `BV037*` 会连 `BV0370-…` 一起算进来，中文标题的素材又一个都找不到。
    video = None
    for v in paths.videos_for(slug):
        try:
            video = {"name": v.name, "path": str(v), "bytes": v.stat().st_size}
        except OSError:
            pass
        break

    parts = {
        "processed": {"bytes": proc_bytes, "files": proc_files},
        "outputs": {"bytes": out_bytes, "files": out_files},
        "funscripts": {"bytes": fs_bytes, "files": fs_files},
        "draft": {"bytes": draft_bytes, "files": draft_files},
    }
    return {
        "slug": slug,
        "parts": parts,
        "products": sum(p["bytes"] for p in parts.values()),
        "video": video,
    }


def busy(slug: str) -> bool:
    """这条素材有没有在跑/排队的作业。"""
    from . import jobs
    return any(j.slug == slug and j.state in ("queued", "running")
               for j in jobs.all_jobs())


def delete(slug: str, with_video: bool = False) -> dict:
    """删掉一条素材的产物；`with_video=True` 连原片一起。

    返回回执：删了什么（removed）、有什么没删掉（failed）、回收了多少字节。
    界面直接把这两条显示出来。
    """
    if not SAFE_SLUG.match(slug):
        raise ValueError(f"素材名不合法：{slug!r}")
    if busy(slug):
        raise RuntimeError("这条素材还有作业在跑（或排队），先等它结束或取消")

    freed = 0
    removed: list[str] = []
    failed: list[str] = []

    # ---- 原片先删，而且**删不掉就整件事停下** ----
    #
    # 顺序是有意的。原片是唯一"会被别的程序占着"的东西（浏览器正在流式播它），
    # 而 Windows 上被占用就是删不掉、没有强制办法。放最后删的话，异常一冒泡，
    # 产物已经没了——留下"产物删干净、原片还在"的半删状态，用户看到一条点不开
    # 也没产物的素材，比直接失败更难理解。放最前面失败就等于什么都没做。
    if with_video:
        # 按规则反算，不是前缀匹配——这行是**删除**，`{slug}*` 会把
        # `BV0370-…` 这种前缀撞车的素材一起删掉（见 paths.videos_for）。
        vids = paths.videos_for(slug)
        if vids:
            n = sum(v.stat().st_size for v in vids if v.exists())
            for v in vids:
                try:
                    _unlink_locked(v)
                except OSError as exc:
                    raise RuntimeError(
                        f"原片被别的程序占用，删不掉：{v.name}。"
                        "最常见的原因是浏览器还在播它（或者刚播过、连接还没断）。"
                        "先暂停播放、或切到别的素材，再删一次；"
                        "也可以取消勾选「连原片一起删」，只删产物。"
                        f"（{exc}）") from exc
            freed += n
            removed.append(f"原片 {len(vids)} 个（{_mb(n)}）")

    targets = [
        (paths.OUTPUTS / slug, "重建产物"),
        (paths.FUNSCRIPTS / slug, "脚本"),
        (convert.draft_dir(slug), "脚本草稿"),
    ]
    for d, label in targets:
        if d.is_dir():
            n, _ = _tree_size(d)
            shutil.rmtree(d, ignore_errors=True)
            freed += n
            removed.append(f"{label}（{_mb(n)}）")

    # 预处理产物是散文件，按 glob 删。**它也可能被占用**（播放画面切到
    # "预处理产物"那一档时），所以逐个容错：删不掉的记下来报给用户，
    # 不要因为一个文件让整件事变成 500。
    proc = sorted(paths.PROCESSED.glob(f"{slug}_*p*.mp4"))
    if proc:
        n, gone = 0, 0
        for p in proc:
            sz = p.stat().st_size if p.exists() else 0
            try:
                _unlink_locked(p, attempts=3)
                n += sz
                gone += 1
            except OSError:
                failed.append(p.name)
        freed += n
        removed.append(f"预处理视频 {gone} 个（{_mb(n)}）")

    # 内存里的缓存必须跟着清，否则界面还会拿旧数据（比如刚删完又能"打开"）
    convert.clear(slug)
    motion.clear(slug)
    index.scan()

    return {"slug": slug, "removed": removed, "failed": failed,
            "bytes": freed, "with_video": with_video}


def _unlink_locked(p: Path, attempts: int = 4, delay: float = 0.15) -> None:
    """删一个可能正被别的进程占用的文件，先短重试几次。

    Windows 上文件被打开着就是删不掉（WinError 32），而且**没有**强制删除的办法
    ——MoveFileEx 那种"重启后再删"只会把问题推迟到用户看不见的地方，更糟。
    好在占用多半是瞬时的（播放器刚切走、HTTP 连接刚断），所以重试几次能过掉
    大部分情况；还不行就抛出去，由调用方给一句能指路的话。
    """
    last: OSError | None = None
    for _ in range(max(1, attempts)):
        try:
            p.unlink(missing_ok=True)
            return
        except OSError as exc:                       # noqa: PERF203
            last = exc
            time.sleep(delay)
    raise last if last else OSError(f"删不掉 {p}")


def _mb(n: int) -> str:
    if n >= 2**30:
        return f"{n / 2**30:.2f} GB"
    if n >= 2**20:
        return f"{n / 2**20:.0f} MB"
    return f"{max(1, n // 1024)} KB"     # 小文件别显示成 "0 MB"
