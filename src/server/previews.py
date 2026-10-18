"""渲染预览：目录、命名、参数校验、元数据。

**产物落在 `workspace/outputs/<slug>/`，和 `quad.mp4` 并列**——它是给人看的
正式产物，不是可以随手清掉的缓存（早先放 `web_cache` 是因为一开始把它当草稿，
实际用起来这些正是"我要留着对照"的东西）。同样参数再渲一次会用同名文件覆盖。

每个 mp4 旁边放一个同名 `.json` 记参数。界面要靠它知道两件事：
  · 这条预览是**哪几格**拼的（列表上得看得懂，不然一堆 hash 文件名没法挑）
  · 它覆盖**哪一段**（`start_ms`）。全段的预览可以直接顶替主画面，因为和主
    时间轴同基；只渲了一段的话，主画面得把播放位置减掉这个偏移才对得上。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from . import paths

FPS = 30.0                      # 这条管线固定 30fps（见 batch_run 的说明）
MAX_CELLS = 12                  # 格子数的上限：每格一次 mesh 栅格化，且画布线性变宽
DEFAULT_VIEWS = "raw,overlay,world@0,world@90"   # 等于旧的 quad


def output_dir(slug: str) -> Path:
    """这条素材的输出目录——预览和 quad.mp4 都放这儿。

    注意别在这里 glob `*.mp4`：目录里还有 quad.mp4（管线第 6 步的产物）。
    预览的文件名里一定带 `__<8位哈希>`，认这个。
    """
    return paths.OUTPUTS / slug


def _kinds() -> tuple[str, ...]:
    """视图白名单。**从 render_overlay 拿**，不在这里再抄一份——
    两处各存一份的表现是"界面点得动、渲染报未知视图"。"""
    from render_overlay import CELL_KINDS
    return CELL_KINDS


def normalize(body: dict) -> tuple[dict | None, str | None]:
    """校验并规整前端传来的参数。返回 (params, 错误说明)。"""
    views = (body.get("views") or DEFAULT_VIEWS)
    if not isinstance(views, str):
        return None, "views 要是逗号分隔的字符串"
    cells = [t.strip() for t in views.split(",") if t.strip()]
    if not cells:
        return None, "一个视图都没选"
    if len(cells) > MAX_CELLS:
        return None, f"最多 {MAX_CELLS} 格（再多画布宽得没法看，渲染也慢）"
    kinds = _kinds()
    for tok in cells:
        kind, _, yaw = tok.partition("@")
        if kind.strip().lower() not in kinds:
            return None, f"未知视图 {kind!r}，可选：{', '.join(kinds)}"
        if yaw.strip():
            try:
                float(yaw)
            except ValueError:
                return None, f"{tok!r} 的朝向不是数字"

    grid = (body.get("grid") or "").strip() or None
    if grid:
        m = re.fullmatch(r"([1-6])x([1-6])", grid.lower())
        if not m:
            return None, "grid 要写成 RxC，如 2x2（1~6）"
        r, c = int(m.group(1)), int(m.group(2))
        if r * c < len(cells):
            return None, f"{len(cells)} 格放不进 {r}x{c} 网格"

    cell_h = body.get("cell_h")
    if cell_h in (None, "", "auto"):
        cell_h = None
    else:
        try:
            cell_h = int(cell_h)
        except (TypeError, ValueError):
            return None, "cell_h 要是数字"
        if not 120 <= cell_h <= 2160:
            return None, "cell_h 要在 120~2160 之间"

    frames = (body.get("frames") or "").strip() or None
    if frames:
        m = re.fullmatch(r"(\d*):(\d*)", frames)
        if not m or not (m.group(1) or m.group(2)):
            return None, "frames 要写成 I0:I1（如 372:477，一端可省）"
        a = int(m.group(1) or 0)
        b = int(m.group(2)) if m.group(2) else None
        if b is not None and b <= a:
            return None, "结束帧要大于起始帧"

    return {"views": ",".join(cells), "grid": grid, "cell_h": cell_h,
            "frames": frames}, None


def _start_frame(frames: str | None) -> int:
    return int((frames or "0:").split(":")[0] or 0)


def make_name(params: dict) -> str:
    """文件名带上视图（给人看）+ 参数哈希（同样参数重复提交就复用同名）。"""
    tag = re.sub(r"[^A-Za-z0-9]+", "-", params["views"]).strip("-")[:40]
    key = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return f"{tag or 'preview'}__{hashlib.sha1(key.encode()).hexdigest()[:8]}.mp4"


def meta_path(video: Path) -> Path:
    return video.with_suffix(".json")


def write_meta(video: Path, params: dict) -> dict:
    """渲染**成功之后**才写。

    提交时就写的话，渲染失败或被中断会留下一条"有元数据的空文件"，列表把它
    当正常产物列出来，点开是个坏视频——本项目的预览就这么出现过一条 0 字节的
    （中断渲染留下的），而且看不出它是坏的。宁可不显示。
    """
    start = _start_frame(params["frames"])
    meta = {**params, "start_ms": round(start / FPS * 1000, 3),
            "full": params["frames"] is None, "fps": FPS,
            "created": time.time()}
    meta_path(video).write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    return meta


# 小于这个就算残骸。一帧的 mp4 也有几 KB，所以 1KB 只会是"ffmpeg 一个字节都没写"。
MIN_BYTES = 1024

# 预览的文件名：`<视图串>__<8位哈希>.mp4`。**必须靠它和 quad.mp4 区分开**——
# 两者现在同目录，只按 `*.mp4` 认的话，列表里会多出一条"quad.mp4 预览"，
# 而 DELETE 路由还能把正式 quad 删掉。
NAME_RE = re.compile(r"^[A-Za-z0-9@\-]+__[0-9a-f]{8}\.mp4$")


def is_preview_name(name: str) -> bool:
    return bool(NAME_RE.match(name))


def _usable(p: Path) -> bool:
    return p.is_file() and p.stat().st_size >= MIN_BYTES


def _path_raw(slug: str, name: str) -> Path | None:
    """按名字定位预览文件，不看大小。**名字必须是纯文件名且长得像预览**——
    这个值来自 URL，带上 `/` 或 `..` 就能读到目录外的文件。"""
    if name != Path(name).name or not is_preview_name(name):
        return None
    p = output_dir(slug) / name
    return p if p.is_file() else None


def path_of(slug: str, name: str) -> Path | None:
    """给**播放**用的：残骸（0 字节那种）不当成预览，返回 None 让路由 404。"""
    p = _path_raw(slug, name)
    return p if p is not None and _usable(p) else None


def count(slug: str) -> int:
    """这条素材有几条可用预览。

    单独一个函数是为了**不读元数据**：素材列表每轮询一次就要问所有素材，
    `list_previews` 会把每个 sidecar json 都读一遍，纯浪费。
    """
    d = output_dir(slug)
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.mp4")
               if is_preview_name(p.name) and _usable(p))


def list_previews(slug: str) -> list[dict]:
    """列出可用的预览，新的在前。

    残骸（0 字节 / 太小）直接跳过：那是渲染被中断留下的，元数据可能还在，
    列出来只会让人点到一个空视频。同参数的再次渲染会用同一个文件名覆盖它。
    """
    d = output_dir(slug)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        if not is_preview_name(p.name) or not _usable(p):
            continue      # quad.mp4 和渲染残骸都不算预览
        st = p.stat()
        try:
            meta = json.loads(meta_path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # 元数据缺失不该让整页打不开（老产物、手删了一半都算）
            meta = {"views": "?", "start_ms": 0.0, "full": True}
        out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime, **meta})
    return out


def delete(slug: str, name: str) -> bool:
    """删一条预览，连同它的元数据。

    用 _path_raw 而不是 path_of：中断留下的 0 字节残骸也要能删掉——列表里
    看不见它，但清理得干净。
    """
    p = _path_raw(slug, name)
    if p is None:
        return False
    meta_path(p).unlink(missing_ok=True)
    p.unlink()
    return True
