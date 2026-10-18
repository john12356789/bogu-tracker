"""工作区路径与素材命名约定。

命名规则**只在 batch_run.ascii_slug 里定义一次**，这里直接导入它，
不另写一份——两套命名规则一旦漂移，界面上会找不到产物，而且很难查。

为什么需要 ASCII 短名：GVHMR 用 hydra 解析命令行覆盖项，而 hydra 不接受非 ASCII，
中文名会让它直接抛 LexerNoViableAltException。所以"素材名"在磁盘上有两种形态：
  · 原始名 `BV16Vd3BBEZA-小奶糖Candy-可御可萝.mp4`  —— 给人看的
  · 短名   `BV16Vd3BBEZA`                            —— 给目录名用的
两者通过 ascii_slug() 单向映射。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent      # src/web
SRC = WEB.parent                           # src
ROOT = SRC.parent                          # 项目根

# batch_run 就在 src/ 下；to_funscript.py 也靠 sys.path 里有 src/ 才能 import viz.schema
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batch_run import PY_GVHMR, PY_MAIN, ascii_slug  # noqa: E402

WORKSPACE = ROOT / "workspace"
VIDEOS = WORKSPACE / "videos"
PROCESSED = WORKSPACE / "processed"
OUTPUTS = WORKSPACE / "outputs"
FUNSCRIPTS = WORKSPACE / "funscripts"
CACHE = WORKSPACE / "web_cache"

# 六个 TCode 轴 -> funscript 文件名后缀。None 表示主轴（无后缀）。
# 与 to_funscript.py 的 SUFFIX 一致，也与 MultiFunPlayer 的分区约定一致。
AXIS_SUFFIX: dict[str, str | None] = {
    "L0": None, "L1": "surge", "L2": "sway",
    "R0": "twist", "R1": "roll", "R2": "pitch",
}

# 重建结果的坐标系。**名字在这里定义一份**（语义说明在 convert.SPACES，
# 那边讲的是"两者不是同一动作的两种表示"）：导出可以只导其中一个，所以"有没有
# 结果"、"有哪些坐标系"都要按这一组名字去盘上找，而不是写死 incam。
SPACES = ("incam", "global")
AXES = list(AXIS_SUFFIX)


def slug_of(video_path: Path) -> str:
    """原片 -> ASCII 短名。"""
    return ascii_slug(video_path.stem)


def videos_for(slug: str, root: Path | None = None) -> list[Path]:
    """短名 -> 原片。**按规则反算，不做前缀匹配**。

    前缀 glob（`VIDEOS.glob(f"{slug}*.mp4")`）有两处会咬人：

      · `BV037*` 会把 `BV0370-另一个素材.mp4` 一起匹配上——而调用它的地方是
        **删除**路径（cleanup.delete 的"连原片一起删"），那就是删错文件；
      · slug 只在"名字开头是 ASCII"时才是文件名的前缀。中文标题的素材落到哈希
        兜底（见 batch_run.ascii_slug），前缀匹配一个都找不到——删除时静默少删
        一个文件。实测 `旗袍—卡点爆摇.mp4` 就是这样。

    所以反过来算一遍：把目录里每条原片按 slug_of() 跑一次，相等的就是它。
    50 来个文件，代价可以忽略。

    返回列表而不是单个：slug 理论上可以撞（两条素材开头 ASCII 段相同），
    返回列表让调用方自己决定是"取第一个"还是"全都处理"。
    """
    d = VIDEOS if root is None else root
    if not d.is_dir():
        return []
    return [p for p in sorted(d.glob("*.mp4")) if slug_of(p) == slug]


# 预处理产物的可读性缓存：Path -> (大小, mtime, 能不能读)。
# 这个函数在列表页上是**每请求 × 每条素材**各调一次，每次都跑 ffprobe 会拖慢
# 整个列表（56 条素材就是 56 次进程）。文件变了（大小/mtime）缓存自动失效，
# 而编码产物写完之后就不再变，所以实际上一个进程里只探测一次。
_READABLE: dict[Path, tuple[int, float, bool]] = {}


def _readable_cached(p: Path, check) -> bool:
    try:
        st = p.stat()
    except OSError:
        return False
    hit = _READABLE.get(p)
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime:
        return hit[2]
    ok = check(p)
    _READABLE[p] = (st.st_size, st.st_mtime, ok)
    return ok


def processed_video(slug: str) -> Path | None:
    """预处理产物。高度可配，所以按 glob 找；取最新的**能读的**一个。

    注意：2D 关键点是在**这个**视频的像素空间里算出来的，
    叠加必须画在它上面，不能用原片。

    **必须验能不能读**。MP4 的索引在文件末尾，编码被打断就留下大小正常但谁也
    读不了的残件，而这里过去只看文件在不在——于是残件被当成"预处理已完成"：
    列表上第 1 步亮着，点跑会被跳过，第 2 步一打开就报 pyav 的错误。跳过读不了
    的那个，整条链就自洽了（第 1 步显示未完成 -> 点跑会重编）。
    """
    if not PROCESSED.is_dir():
        return None
    import preprocess                     # 纯标准库模块，不会和这里的导入绕圈
    cands = sorted(PROCESSED.glob(f"{slug}_*p*.mp4"), key=lambda p: p.stat().st_mtime)
    for p in reversed(cands):             # 从最新的往回找，取第一个能读的
        if _readable_cached(p, preprocess.video_readable):
            return p
    return None


def gvhmr_dirs(slug: str) -> list[Path]:
    """GVHMR 的输出目录（可能有多套检测器，如 gvhmr_rtmpose / gvhmr_vitpose）。"""
    root = OUTPUTS / slug
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("gvhmr_"))


# 多人素材里，除主轨迹之外的每条轨迹各占一个独立的输出根：
#   outputs/<slug>/gvhmr_rtmpose/<stem>/          ← 主轨迹（一直以来的那个，零迁移）
#   outputs/<slug>/gvhmr_rtmpose__t<tid>/<stem>/  ← 其余轨迹
# 为什么是"多一个目录"而不是"多一个文件名"：三个步骤脚本本来就按输出目录参数化
# （prep_boxes --output-root / run_rtmpose --src,--dst / run_gvhmr --output_root），
# 换个根目录就等于换个 track，脚本内部一行都不用动。
TRACK_DIR_RE = re.compile(r"^gvhmr_.+__t(?P<tid>\d+)$")


def track_dirs(slug: str) -> list[tuple[int, Path]]:
    """其余轨迹的 (tid, 输出根目录)，按 tid 排。"""
    root = OUTPUTS / slug
    if not root.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in sorted(root.iterdir()):
        m = TRACK_DIR_RE.match(p.name) if p.is_dir() else None
        if m:
            out.append((int(m.group("tid")), p))
    return out


def rt_root(slug: str, tid: int | None = None) -> Path | None:
    """轨迹的"输出根"（与 batch_run 的 ctx["rt_root"] 一致）。

    tid=None 给主轨迹的输出根；给 tid 时先找 `gvhmr_*__t<tid>`，找不到就返回 None
    （而不是拿主轨迹顶上——那会把两个人的关节数据悄悄搞混）。
    """
    if tid is None:
        dirs = gvhmr_dirs(slug)
        # 主轨迹 = 不带 __t 后缀的那个。**必须显式排除带后缀的**，否则
        # sorted 的顺序会让人误以为第一个永远是主轨迹。
        plain = [d for d in dirs if not TRACK_DIR_RE.match(d.name)]
        return plain[0] if plain else (dirs[0] if dirs else None)
    for t, d in track_dirs(slug):
        if t == tid:
            return d
    return None


def results_dir(slug: str, tid: int | None = None) -> Path | None:
    """含重建结果 npz 的那个目录（**任一坐标系**都算）。

    `tid=None` 时**优先主轨迹**：`gvhmr_dirs` 的 glob 会把 `__t<tid>` 那些也捞进来，
    不排除的话多人素材上"当前素材的重建结果"会随机落到某个人身上。

    判据是"任一坐标系"而不是写死 incam：导出可以只导世界系（见 batch_run 里
    「导出 npz」的坐标系选项），写死 incam 的话那份产物在整条链路上等于不存在
    ——列表说这条素材没重建结果、算脚本说找不到 npz，全都指向一个不存在的错。
    """
    root = rt_root(slug, tid)
    if root is None:
        return None
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and any((sub / f"hmr4d_results_{s}.npz").exists() for s in SPACES):
            return sub
    return None


def spaces_on_disk(slug: str, tid: int | None = None) -> list[str]:
    """这条素材（这条轨迹）**真的**有哪些坐标系的重建结果。

    界面用它把「生成来源」里没有的那一档置灰：允许点到一档算出 result 里没有的
    东西，报出来的一定是"没有 hmr4d_results_global.npz"这种离用户操作很远的错。
    """
    d = results_dir(slug, tid)
    if d is None:
        return []
    return [s for s in SPACES if (d / f"hmr4d_results_{s}.npz").exists()]


def primary_tid(slug: str) -> int | None:
    """主轨迹的 track id（= bbx_all.json 里第一条）。

    它的产物**不带 `__t` 后缀**（一直以来的那个目录），所以"用户明确选了主轨迹"
    这件事需要靠这个 id 认出来——不认的话会去 `gvhmr_rtmpose__t<主tid>` 找一个
    根本不存在的目录，然后报"这条轨迹还没有重建结果"。
    """
    p = people_json(slug)
    if p is None:
        return None
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
        segs = info.get("segments") or []
        return int(segs[0]["tid"]) if segs else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def npz_for_track(slug: str, space: str, tid: int | None) -> Path | None:
    """某条轨迹某个坐标系的重建结果。tid=None 或等于主轨迹 id 都是主轨迹那份。"""
    d = results_dir(slug, tid)
    if d is None and tid is not None and tid == primary_tid(slug):
        d = results_dir(slug, None)
    if d is None:
        return None
    p = d / f"hmr4d_results_{'incam' if space == 'incam' else 'global'}.npz"
    return p if p.exists() else None


def merge_json(slug: str) -> Path:
    """人物归并的分组决策（人工在界面上确认过的）。

    和 cuts.json 一样放在素材根下，**不放进任何输出根**：它描述的是"哪几条轨迹
    属于同一个人"，和用哪次预处理、哪种检测器重跑都无关。文件不存在 = 还没有
    决策 = 归并这一步什么都不做（见 batch_run 的 step_merge）。
    """
    return OUTPUTS / slug / "people_merge.json"


def people_json(slug: str) -> Path | None:
    """轨迹段清单（bbx_all.json，prep_boxes 写的）。没有就是单人素材。"""
    root = rt_root(slug)
    if root is None:
        return None
    for sub in sorted(root.iterdir()):
        f = sub / "bbx_all.json"
        if f.exists():
            return f
    return None


def rt_subdirs(slug: str) -> list[Path]:
    """GVHMR 的 rt_dir 们：outputs/<slug>/gvhmr_<检测器>/<预处理产物名>/。

    名字里带预处理产物的主干，所以同一条素材重跑不同分辨率会留下多个。见
    step_products 里为什么不去挑是哪一个。
    """
    out: list[Path] = []
    for d in gvhmr_dirs(slug):
        try:
            out.extend(p for p in sorted(d.iterdir()) if p.is_dir())
        except OSError:                      # 刚好被删/正在写，跳过就好
            continue
    return out


def step_products(slug: str) -> dict[str, bool]:
    """步骤 id -> 该步的产物在盘上有没有。键与 batch_run.STEP_ORDER 一一对应。

    用途只有一个：判断"只勾后面几步行不行"——盘上有前置的产物，就等于前置
    已经满足（见 batch_run.check_prereqs）。

    标志物和 batch_run 里各步实际写出来的东西一一对应：
        boxes    rt_dir/preprocess/bbx.pt
        rtmpose  rt_dir/preprocess/vitpose.pt
        gvhmr    rt_dir/hmr4d_results.pt
        export   rt_dir/hmr4d_results_<坐标系>.npz（任一坐标系）
        render   outputs/<slug>/quad.mp4
    boxes/rtmpose/gvhmr/export 都在 rt_dir 下，而**不挑是哪个 rt_dir**：与
    results_dir() 的宽松度保持一致，否则会出现"列表标着'有重建'、勾选面板却说
    '缺前置'"这种自相矛盾。哪个 rt_dir 属于哪次预处理没记在文件名里，挑也没法挑。
    """
    subs = rt_subdirs(slug)

    def in_rt(*parts: str) -> bool:
        return any(s.joinpath(*parts).exists() for s in subs)

    return {
        "preprocess": processed_video(slug) is not None,
        "boxes": in_rt("preprocess", "bbx.pt"),
        "rtmpose": in_rt("preprocess", "vitpose.pt"),
        "gvhmr": in_rt("hmr4d_results.pt"),
        # 导出：**任一坐标系**有就算跑过（可以只导世界系，见 batch_run.STEP_OPTS）。
        # 写死 incam 的话，只导了世界系的素材会被当成"没导出"，而它的产物明明在。
        "export": any(in_rt(f"hmr4d_results_{s}.npz") for s in SPACES),
        # 归并：标志物是**这一根的分组凭证**（batch_run 的归并步骤写的那份小
        # JSON），不是 npz——npz 和「多人轨迹」出来的那份长得一样，分不出来。
        # 凭证只在这一组真的跑完之后才写，所以它既是"跑过"也是"跑的是哪几段"。
        "merge": any((d / "merge_spec.json").exists() for _, d in track_dirs(slug)),
        # 其余轨迹：**任何一条**跑出了 npz 就算有产物。不挑 tid、不挑坐标系——
        # 这个圆点回答的是"这一步跑过没有"，和上面几条同一个宽松度。
        "tracks": any(results_dir(slug, tid) is not None for tid, _ in track_dirs(slug)),
        "render": quad_path(slug) is not None,
    }


def funscript_dir(slug: str) -> Path | None:
    d = FUNSCRIPTS / slug
    return d if d.is_dir() else None


def quad_path(slug: str) -> Path | None:
    """四格预览视频（原片 | 叠加 | 正面 | 侧面），在 output 根下。"""
    p = OUTPUTS / slug / "quad.mp4"
    return p if p.exists() else None


def cuts_json(slug: str) -> Path:
    """切点清单（scene_cuts.py 出的那份）。

    放在 outputs/<slug>/ 下而不是 web_cache：它是**这条素材的派生数据**，
    而删除素材时 cleanup.delete 会整目录 rmtree 掉 outputs/<slug>——
    不用再单独记一笔，也就不会漏删。
    """
    return OUTPUTS / slug / "cuts.json"


def funscript_files(slug: str) -> dict[str, Path]:
    """轴名 -> funscript 路径（正式产物目录）。"""
    d = funscript_dir(slug)
    return funscript_files_in(d) if d else {}


def funscript_files_in(d: Path) -> dict[str, Path]:
    """轴名 -> funscript 路径（任意目录）。

    主轴靠"文件名只有一个点"来认：`X_1280p.funscript` 是主轴，
    `X_1280p.pitch.funscript` / `.multi.funscript` 都有两个点。
    不用字母序碰运气——那只是恰好排对而已。
    """
    if d is None or not Path(d).is_dir():
        return {}
    all_fs = sorted(Path(d).glob("*.funscript"))
    out: dict[str, Path] = {}
    for axis, suffix in AXIS_SUFFIX.items():
        if suffix is None:
            cands = [p for p in all_fs if p.name.count(".") == 1]
        else:
            cands = [p for p in all_fs if p.name.endswith(f".{suffix}.funscript")]
        if cands:
            out[axis] = cands[0]
    return out
