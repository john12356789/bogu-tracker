"""素材索引：扫描 workspace，列出有哪些片子、各自算到哪一步了。

**这个模块绝不读视频或 npz 的字节。** 原片最大 671 MB、npz 273 MB，
18 条加起来好几个 GB——列表页只需要"有没有、多大、多长"，
所以这里只用 stat() 和 ffprobe（ffprobe 只读文件头，不解码）。

ffprobe 的结果按 (size, mtime) 缓存到 workspace/web_cache/index.json，
文件没变就不重跑。这是让列表秒开的关键。
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import previews
from .paths import (
    AXES,
    CACHE,
    VIDEOS,
    funscript_files,
    processed_video,
    quad_path,
    results_dir,
    slug_of,
    spaces_on_disk,
    step_products,
)

_LOCK = threading.Lock()
_CACHE_FILE = CACHE / "index.json"


@dataclass
class Clip:
    slug: str
    stem: str                    # 原始文件名主干（含中文），给人看
    filename: str
    size: int                    # 字节
    mtime: float
    duration: float | None = None   # 秒
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    has_processed: bool = False
    has_results: bool = False
    has_quad: bool = False
    # 步骤 id -> 该步产物在不在。上面三个 has_* 是给人看的标签，这个是给
    # 勾选面板用的逐步骤细粒度判断（同一个事实在 paths.step_products 里只算一次）。
    products: dict[str, bool] = field(default_factory=dict)
    # 渲染预览的条数。列表上要显示"这条素材有几条预览"，而预览是按文件摊在
    # outputs/<slug>/ 里的（和 quad.mp4 同目录），只能现数。
    n_previews: int = 0
    axes: list[str] = field(default_factory=list)
    # 这条素材**真的**有哪些坐标系的重建结果（导出可以只导相机系，见
    # batch_run.STEP_OPTS）。界面据此把「生成来源」里没有的那一档置灰——
    # 允许点到不存在的那一档，报出来的是"没有 hmr4d_results_global.npz"，
    # 离用户那一下点击很远。
    spaces: list[str] = field(default_factory=list)
    probe_error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_CACHE_FILE)   # 原子替换，避免半截文件


def probe(path: Path) -> dict:
    """ffprobe 只读文件头。失败返回 {}，不抛——一条坏片不该拖垮整个列表。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,width,height:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        ).stdout
        d = json.loads(out or "{}")
        st = (d.get("streams") or [{}])[0]
        num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else None
        return {
            "duration": float(d.get("format", {}).get("duration") or 0) or None,
            "fps": fps or None,
            "width": st.get("width"),
            "height": st.get("height"),
        }
    except Exception as exc:                      # noqa: BLE001 — 兜底就是它的职责
        return {"probe_error": str(exc)[:200]}


def scan(use_cache: bool = True) -> list[Clip]:
    """扫描 workspace/videos，返回全部素材。按文件名排序，稳定。"""
    if not VIDEOS.is_dir():
        return []

    with _LOCK:
        cache = _load_cache() if use_cache else {}
        dirty = False
        clips: list[Clip] = []

        for path in sorted(VIDEOS.glob("*.mp4")):
            try:
                st = path.stat()
            except OSError:
                continue
            slug = slug_of(path)
            key = path.name
            hit = cache.get(key)
            if hit and hit.get("size") == st.st_size and hit.get("mtime") == st.st_mtime:
                meta = {k: hit.get(k) for k in
                        ("duration", "fps", "width", "height", "probe_error")}
            else:
                meta = probe(path)
                cache[key] = {"size": st.st_size, "mtime": st.st_mtime, **meta}
                dirty = True

            axes = set(funscript_files(slug))
            # 上面三个 has_* 和这里读的是同一批文件（processed_video / results_dir
            # / quad_path 各自 stat 一次，再 glob 一次）。多花的只是几十次 stat：
            # 列表页要用的元信息本来就只准 stat 和 ffprobe。
            prods = step_products(slug)
            clips.append(Clip(
                n_previews=previews.count(slug),
                slug=slug, stem=path.stem, filename=path.name,
                size=st.st_size, mtime=st.st_mtime,
                has_processed=prods["preprocess"],
                has_results=prods["export"],
                has_quad=prods["render"],
                products=prods,
                axes=[a for a in AXES if a in axes],
                spaces=spaces_on_disk(slug),
                **meta,
            ))

        if dirty:
            _save_cache(cache)
        return clips


def by_slug(slug: str) -> Clip | None:
    return next((c for c in scan() if c.slug == slug), None)
