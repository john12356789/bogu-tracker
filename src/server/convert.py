"""参数重算：把 to_funscript 的参数变成一次毫秒级的重新生成。

**数学只有一份实现**——直接调 to_funscript.run()，不在这里重写任何公式。
理由：如果 TS 和 Python 各有一套，界面上调出来的结果会和 batch 产出的 funscript
对不上，而这类不一致极难排查。

两个关键设计：

1. **seq 常驻缓存**。贵的是读 npz（见 sources.from_smplx_npz 的 with_mesh 说明），
   便宜的是算。缓存之后改一次参数是 0.25s 上下，而不是重新读盘。

2. **写到 preview 目录，不覆盖正式产物**。正式产物在 workspace/funscripts/<slug>/，
   是批处理的输出。调参是个反复试的过程，直接盖掉它会把能用的版本弄丢。
   调好了再由 /promote 显式提交。
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import tempfile
import threading
from pathlib import Path

from . import paths  # 必须先 import：它负责把 src/ 放进 sys.path

import to_funscript  # noqa: E402
import tracklet_merge  # noqa: E402
from viz import sources  # noqa: E402

_LOCK = threading.Lock()
_SEQ: dict[str, object] = {}      # slug -> MotionSequence（无 mesh，约 1 MB）

# 允许前端覆盖的参数。白名单，不直接把请求里的键拼成命令行。
_ALLOWED: dict[str, type] = {
    "fps": float, "highpass": float, "highpass_rot": float,
    "travel_l0": float, "travel_l1": float, "travel_l2": float,
    "max_deg": float, "neutral": str, "order": str, "flip": str,
    "smooth": int, "fit": float, "de_bias": bool, "rate": float,
    # 按时间窗局部归一化的窗长（秒）。0 = 全段共用一个比例尺。
    # 独立于 fit：正常与 fit>0 同用（窗内用同一个分位），
    # fit=0 时回落到 99% 分位（to_funscript 的 WIN_PCT_FALLBACK），照样生效。
    "fit_window": float,
    # 参考点：「阴部跟踪」相对「骨盆跟踪」多一根杆臂
    "reference": str, "ref_fwd": float, "ref_down": float,
    # 跨切点处理（切点清单本身不走这里，后端按 slug 找 paths.cuts_json）
    "cut_reset": bool, "cut_blank": float, "cut_scale": bool,
    # 后处理（输出前）：A 高度相关限位、B 位移→旋转补偿。**两个开关默认关**，
    # 关掉时输出与以前逐元素相同（回归见 test_postprocess）。
    "height_limit": bool, "hl_start": float, "hl_top": float,
    "compensate": bool, "comp_pitch": float, "comp_roll": float,
    # 逐轴倍乘（6 条轴各自一个系数，1 = 不变）
    "gain_l0": float, "gain_l1": float, "gain_l2": float,
    "gain_r0": float, "gain_r1": float, "gain_r2": float,
    # 后处理的执行顺序（界面拖动排出来的）。字符串，to_funscript 里解析成 id 列表，
    # 认不出的丢掉、缺的补在末尾。
    "post_order": str,
}


def defaults() -> dict:
    """界面上要显示的参数默认值。

    **从 argparse 自身的定义取，不在这里列清单**——之前是手写的键名列表，
    加了 --reference / --ref-* 之后它就会被漏掉，界面上那几项永远显示空值。
    """
    ns = to_funscript.parse_args(["x", "--out", "y"])
    return {k: getattr(ns, k) for k in _ALLOWED if hasattr(ns, k)}


def to_argv(params: dict) -> list[str]:
    """请求 JSON -> to_funscript 的命令行参数。白名单之外的键直接丢弃。"""
    out: list[str] = []
    for key, value in params.items():
        if key not in _ALLOWED or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
        else:
            out += [flag, str(value)]
    return out


def draft_dir(slug: str) -> Path:
    """脚本草稿目录的**路径**，不建目录。

    光想知道"草稿占了多大"（删除确认弹窗）不该在磁盘上留下空目录，
    所以 mkdir 只在真的要往里写的时候做——见 preview_dir。
    """
    return paths.CACHE / "preview" / slug


def preview_dir(slug: str) -> Path:
    d = draft_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wipe(d: Path) -> None:
    """清空目录里的文件（不递归、不删目录）。

    重算一次产出的是**全套六轴**，所以进目录前先把上一次的抹掉。留着的话，
    换个坐标系或换过检测器（产物名前缀来自 npz 所在目录名）就会在同一目录里
    堆出两套名字不同的脚本，而 funscript_files_in 是按后缀匹配取第一个——
    于是"加载的时候只加载了其中一个"，另一个成了看不见的幽灵。
    """
    for p in d.iterdir():
        if p.is_file():
            p.unlink()


# 可选的来源坐标系。**两者不是同一个世界系的两种视角**（实测拟合刚性变换后
# 残差均值 46 mm、最大 322 mm，远大于人体尺度），而且给人的 motion 也不同：
#   global  重力对齐、+Y 朝上。3D 视图用这个。**不带 K / cam_to_world**，
#           所以幅度保真度那几项诊断在它上面算不了（to_funscript 会跳过并说明）。
#   incam   OpenCV 相机系、+Y 朝下。带相机内外参，全部诊断可用。
SPACES = paths.SPACES


def npz_for(slug: str, space: str, tid: int | None = None) -> Path | None:
    """主轨迹（tid=None）或某条指定轨迹的 npz。

    **tid 必须一路传到底**。漏一层的后果不是报错，是"选谁都给你主轨迹的结果"。
    这个坑在两个地方各踩过一次：`motion.load` 和这里。
    """
    if tid is not None:
        return paths.npz_for_track(slug, space, tid)
    d = paths.results_dir(slug)
    if d is None:
        return None
    p = d / f"hmr4d_results_{space}.npz"
    return p if p.exists() else None


def sequence(slug: str, space: str = "incam", tid: int | None = None):
    """取缓存的 MotionSequence。只读 joints / 2D 关键点，不读 307 MB 的 mesh。

    按 (slug, space, **tid**) 分别缓存——坐标系不同、人不同，关节数值都不同，
    共用一份缓存就会串。**tid 漏进缓存键**的后果是"选了别的人，算出来还是原来那条"，
    而且因为结果一模一样，看起来像"重算没反应"，极难往这儿想。
    """
    space = space if space in SPACES else "global"
    key = (slug, space, tid)
    with _LOCK:
        if key in _SEQ:
            return _SEQ[key]
    npz = npz_for(slug, space, tid)
    if npz is None:
        return None
    try:
        seq = sources.from_smplx_npz(str(npz), fps=30.0, with_mesh=False)
    except Exception:                                # noqa: BLE001
        return None
    with _LOCK:
        _SEQ[key] = seq
    return seq


def fps_from_funscript(d: Path) -> float | None:
    """从同目录主轴的 funscript 反推产物帧率。

    产物是「一帧一个 action」写出来的（`at = round(i*1000/fps_out)`），
    所以 `(n-1)*1000/at[-1]` 就是 `fps_out`。

    **这是唯一能精确还原时间基的办法**：tcode 里的 `I` 只有整数毫秒，
    29.97fps 和 30fps 都写成 `I33`，从它俩反推出来的都是 30.303，
    区分不出来——而 29.97 与 30 差 0.1%，在 81s 的素材上是 82ms 的偏移。
    同一次运行写出的 funscript 才是那个"真值"的载体。
    """
    p = paths.funscript_files_in(d).get("L0")
    if p is None:
        return None
    try:
        acts = json.loads(p.read_text(encoding="utf-8"))["actions"]
    except Exception:                                    # noqa: BLE001
        return None
    if len(acts) < 2 or acts[-1]["at"] <= 0:
        return None
    return (len(acts) - 1) * 1000.0 / acts[-1]["at"]


def _tcode_lines(d: Path) -> list[str]:
    """目录里的 tcode 指令行（一帧一行，去掉空行）。没有就是空表。"""
    files = sorted(Path(d).glob("*.tcode.txt")) if Path(d).is_dir() else []
    if not files:
        return []
    return [s.strip() for s in files[0].read_text(encoding="utf-8").splitlines()
            if s.strip()]


def _read_actions(p: Path | None) -> list[dict]:
    """读一个 funscript 的 actions。读不出来（文件没了/坏了）就当没有。"""
    if p is None:
        return []
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))["actions"]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def read_tcode(d: Path, fps: float | None = None) -> dict | None:
    """读目录下的 tcode.txt，返回指令行与帧率。

    行格式（TCode v0.3）：字母 + 1 位轴号 + 值数字，位数即精度。
      `L05000 L15000 ... I33`  →  L0=0.5000，L1=0.5000，间隔 33ms

    **帧率不能从 `I33` 反推**。`I` 只存整数毫秒，而 30fps 的真实帧间隔是
    33.333ms，写进文件只能是 33 —— 于是 `1000/33 = 30.303fps`，**快 1%**。
    这个误差会随时间累积：11.6s 的素材偏 0.12s，268s 的素材偏 **2.7s**，
    表现就是模拟器越跑越跑到曲线前面（时间轴上的播放头还在前一段，
    SR6 已经做到后一段了）。所以**知道真实帧率的调用方必须传进来**
    （`/convert` 就传它生成时用的那个 `--fps`）。

    调用方不知道时，按这个顺序找真值：
      1. **同目录主轴的 funscript** —— 同一次运行写的，精确（见 fps_from_funscript）
      2. 都没有才退回 `round(1000/iv)`。它比不取整严格更接近真值（`I` 的量化
         在整数毫秒上只会往上偏，取整把这个偏差削掉），30fps 的产物上恰好
         还原成 30，但 29.97fps 的会错成 30（0.1%，81s 素材上 82ms）。
         `make_emu_page.py` 用的也是这个。

    这个格式是 osr-emu 的 write() 直接吃的，前端据此驱动 SR6 模拟器。
    """
    lines = _tcode_lines(d)
    if not lines:
        return None
    m = re.search(r"I(\d+)", lines[0])
    iv = int(m.group(1)) if m else 33
    exact = fps if fps else fps_from_funscript(d)
    return {"lines": lines, "interval_ms": iv,
            "fps": float(exact) if exact else float(round(1000.0 / iv))}


def convert(slug: str, params: dict) -> dict:
    """跑一次 to_funscript，返回六轴 funscript + TCode 指令流 + 文本报告。

    `params` 里的 `space`（global|incam）选出用哪份 npz 生成。它不是
    to_funscript 的参数，所以在这里取出来、不往命令行传。两个坐标系给出的
    动作**确实不同**（实测输出逐字节不一样），不是同一结果的两种表示。
    """
    space = params.get("space") if params.get("space") in SPACES else "global"
    npz = npz_for(slug, space)
    if npz is None:
        return {"ok": False, "error": f"{slug} 没有 hmr4d_results_{space}.npz"}

    # ---- 人物选择（多人素材）----
    #
    # 和 `space` 一样，**不是 to_funscript 的参数**：它决定读哪些 npz，
    # 在这里取出来、不往命令行传。两种形态：
    #   person_tid   整条都用这条轨迹 -> 直接换掉位置参数那份 npz
    #   person_segs  [{from, tid}]    -> 写一份时间线交给 --person-timeline，
    #                                    由 to_funscript 在**姿态层**逐段拼
    # 一个人可能是好几条轨迹段（实测两个舞者各被切成 9~12 段还换过号），
    # 所以第二种是必需的，不只是"更细的选法"。
    tid = params.get("person_tid")
    segs = params.get("person_segs")
    timeline = None
    timeline_rng: list[dict] = []
    if isinstance(segs, list) and segs:
        seq0 = sequence(slug, space)          # 主轨迹；只是拿它读总帧数
        n_all = seq0.n_frames if seq0 is not None else None
        if not n_all:
            return {"ok": False, "error": "读不出总帧数，拼不了人物时间线"}
        # 时间线里记的是**轨迹号**，而合过的那几段已经不是"可选的人"了——它们
        # 的重建产物是各自那段碎片（缺的帧靠插值），要用的是合并后的那一份。
        # 所以这里把被合并的 tid 映射到它所属的"人"（组内最小 tid）。
        # 不映射的话，脚本会静默地用碎片的重建结果算——看着跑成功了，用的是半个人。
        # 生效的分组（**默认关闭：只有人在「人物」面板确认过的才算**），
        # 见 tracklet_merge.effective。这里不再读 bbx_all.json——建议是界面上的
        # 东西，不是流水线的输入。
        eff = tracklet_merge.effective(tracklet_merge.load(paths.merge_json(slug)))
        absorbed = {int(t): g["gid"] for g in eff for t in g["tids"]}
        ordered = sorted(segs, key=lambda x: int(x.get("from") or 0))
        rng: list[dict] = []
        for i, it in enumerate(ordered):
            frm = max(0, int(it.get("from") or 0))
            to = int(ordered[i + 1].get("from") or 0) if i + 1 < len(ordered) else n_all
            t = it.get("tid")
            t = absorbed.get(int(t), int(t)) if t is not None else None
            p = paths.npz_for_track(slug, space, None if t is None else int(t))
            if p is None:
                return {"ok": False,
                        "error": f"轨迹 {t} 还没有 {space} 坐标系的重建结果"}
            rng.append({"from": frm, "to": max(to, frm + 1), "npz": str(p),
                        "tid": None if t is None else int(t)})
        # 时间线文件等 out（预览目录）建好之后再写——out 在本函数后面才定义，
        # 这里先只把区间算出来。（写早了会 UnboundLocalError，而且是在**用户
        # 点了重算之后**才炸，看起来像"重算按钮坏了"。）
        timeline_rng = rng
    elif tid is not None:
        p = paths.npz_for_track(slug, space, int(tid))
        if p is None:
            return {"ok": False, "error": f"轨迹 {tid} 还没有 {space} 坐标系的重建结果"}
        npz = p

    # 整条用同一条轨迹时，序列缓存要按 **tid** 取——缓存键漏了 tid 的话，
    # 选了别人也会命中主轨迹那份，"重算"看着像没反应（因为结果真的完全一样）。
    # 有时段时间线时**必须传 None**：to_funscript 里 `seq` 非空就跳过时间线分支，
    # 那样拼好的时间线会被静默忽略。
    seq = None if timeline_rng else sequence(slug, space, tid)

    # **预览目录不按坐标系分家**：一条素材只有一份脚本草稿。
    # 之前分成 `<slug>__global` / `<slug>__incam` 两份并存，结果就是"两个坐标系
    # 各有一套脚本"——可它们写进正式目录时的**文件名完全一样**（都取自 npz 所在
    # 目录名），谁后提交谁覆盖，界面却按"两套"来理解。现在改成一套：换个坐标系
    # 重算就把上一份覆盖掉，和正式产物的行为一致。
    out = preview_dir(slug)
    # 只算一段（人物面板上每行那个「单独重算」）时**不清目录**：段外那部分正是
    # 这次要原样保留的。清目录留给整条重算——原来的写法是无论如何先 _wipe，
    # 于是"重算某一段"会把其余段全删掉，连 /promote 提交的也成了半截脚本。
    req_range = params.get("range")
    if not req_range:
        _wipe(out)
    argv = [str(npz), "--out", str(out)]
    # 人物时间线落在这个目录里（`out` 在这里才存在，所以写盘放这儿）。
    # 接缝会被 to_funscript 并进 --cuts，所以这份文件同时也是"接缝在哪"的记录。
    if timeline_rng:
        tl_path = out / "person_timeline.json"
        tl_path.write_text(json.dumps(timeline_rng, ensure_ascii=False), encoding="utf-8")
        argv += ["--person-timeline", str(tl_path)]   # 逐段跑时不用它（见下）
    # 切点清单有就带上。**带上本身不改变任何输出**——跨切处理的几个开关默认全关，
    # 清单只是让报告里能列出切点、并让界面上那几个开关有东西可作用。
    cuts_path = paths.cuts_json(slug)
    # 显式给帧率。不给的话 to_funscript 会打一条"未能读到帧率，按 30 假设"的告警——
    # 值恰好也是 30，结果一样，但日志里每次重算都刷一条，且行为变成了隐式的。
    #
    # 判据是 `not params.get("fps")` 而**不是** `"fps" not in params`：前端的
    # paramDefaults 里带着 `fps: null`（argparse 的默认值），键一直都在，
    # 用 `in` 判断就会永远跳过这一条，于是每次都走"按 30 假设"那条路。
    common = to_argv({k: v for k, v in params.items()
                      if k not in ("space", "person_tid", "person_segs", "range")})
    if cuts_path.exists():
        common = ["--cuts", str(cuts_path)] + common

    def _run_one(npz_i: Path, seq_i, argv_i: list[str], out_i: Path):
        """跑一次 to_funscript，返回 (rc, stdout, 解析出来的 args)。

        **把 args 一起带回来**，别在调用方重新 parse 一次：重新 parse 失败时
        （argparse 打参数不合法）会拿到 None，而后面还要用 `args.rate` 去算产物
        帧率——那就是"报的是 A 错，崩的是 B 处"。
        """
        try:
            args_i = to_funscript.parse_args([str(npz_i), "--out", str(out_i)] + argv_i)
        except SystemExit as exc:                    # argparse 校验失败
            return 1, f"参数不合法：{exc}", None
        b = io.StringIO()
        try:
            with contextlib.redirect_stdout(b):
                rc_i = to_funscript.run(npz_i, args_i, seq=seq_i)
        except Exception as exc:                     # noqa: BLE001
            return 1, b.getvalue() + chr(10) + str(exc), args_i
        return rc_i, b.getvalue(), args_i

    buf = io.StringIO()
    args = None
    if timeline_rng:
        # ---- 每段独立成篇 ----
        #
        # 逐段各跑一遍 to_funscript（**各自归一化**），再把产物按时间拼起来。
        # 为什么不在拼接后的序列上跑一次就完事：那样 `--fit` 的分位、行程缩放
        # 都是整条共用的，一段特别离谱的数据会把其余段的幅度一起压扁。
        # 代价是**幅度不再可比**（安静那段会被抬起来），这是用户明确选的。
        #
        # 每段的时间戳是绝对帧号（`--range` 负责），所以拼的时候只管按 at 排序。
        seg_dirs: list[Path] = []
        ok_all = True
        for i, r in enumerate(timeline_rng):
            d_i = out / f"seg{i}"
            d_i.mkdir(parents=True, exist_ok=True)
            npz_i = Path(r["npz"])
            tid_i = r.get("tid")
            seq_i = sequence(slug, space, None if tid_i is None else int(tid_i))
            argv_i = list(common) + ["--range", f"{r['from']}:{r['to']}"]
            if not params.get("fps") and seq_i is not None:
                argv_i += ["--fps", f"{seq_i.fps:g}"]
            buf.write(chr(10) + f"===== 第 {i + 1} 段：帧 {r['from']}~{r['to']}"
                      f"（轨迹 {tid_i}，独立归一化）=====" + chr(10))
            rc_i, out_i, _ = _run_one(npz_i, seq_i, argv_i, d_i)
            buf.write(out_i)
            seg_dirs.append(d_i)
            ok_all = ok_all and rc_i == 0
        # 名字用 **npz.parent.name**（不是 npz.stem）：to_funscript 是
        # `args.name or npz.parent.name`，这里写成 npz.stem 的话，同一次「重算」
        # 会因为这个开关开没开而产出**不同文件名**的同一份脚本。
        _merge_segments(out, seg_dirs, npz.parent.name)
        axes = _read_axes(out)
        return {
            "ok": ok_all, "slug": slug, "space": space,
            "tcode": read_tcode(out, fps=(params.get("rate") or None)),
            "out_dir": str(out), "params": {},
            "axes": axes, "report": buf.getvalue(),
            "segments": timeline_rng,
        }

    # **只传额外参数**：npz 和 --out 由 _run_one 自己加。这里再传一次会让 argparse
    # 看到两个位置参数、直接退出，而 args 变成 None——报错却落在后面用 args.rate
    # 的那一行（"报的是 A 错，崩的是 B 处"）。
    extra = list(common)
    if req_range:
        extra += ["--range", f"{int(req_range[0])}:{int(req_range[1])}"]
    if not params.get("fps") and seq is not None:
        extra += ["--fps", f"{seq.fps:g}"]

    err: str | None = None
    if req_range:
        # ---- 只替换这一段 ----
        #
        # **先算到临时目录，算成功了才动预览目录。** 直接往 out 里写的话，
        # 这一步算挂了（argparse 不过、npz 读不动）预览里就只剩半截脚本，
        # 而 /promote 是把整个预览目录提交成正式产物的——"重算某一段失败"
        # 会变成"其余段被删掉了"。
        seg_tmp = Path(tempfile.mkdtemp(prefix=f"seg_{slug}_"))
        try:
            rc, text, args = _run_one(npz, seq, extra, seg_tmp)
            buf.write(text)
            if rc != 0:
                err = "这一段没算出来，脚本保持原样（没有替换任何一段）"
            else:
                # 整条应该是多少帧：拿**素材自己的**帧数来量，不是拿段来量。
                # `--rate` 重采样时产物帧数和素材帧数本来就不同，那种情况不查
                # （传 None）。
                resampled = bool(args and args.rate and seq is not None
                                 and abs(args.rate - seq.fps) > 1e-6)
                want = None if (resampled or seq is None) else seq.n_frames
                err = _splice_segment(out, seg_tmp, want)
        finally:
            shutil.rmtree(seg_tmp, ignore_errors=True)
    else:
        rc, text, args = _run_one(npz, seq, extra, out)
        buf.write(text)
    axes = _read_axes(out)

    result = {
        "ok": rc == 0 and err is None,
        "slug": slug,
        "space": space,
        # 传**生成时真正用的**帧率，比从 `I33` 反推的准（见 read_tcode 的说明：
        # 反推会快 1%，长素材上累积成好几秒）。产物帧率是 `--rate` 优先、其次
        # `--fps`（to_funscript 里 `fps_out = args.rate else fps`），这里照抄同样的
        # 优先级；两个都没给时交给 read_tcode 取整回落（那是 30，也是默认值）。
        # `if args else None`：argparse 没过时 _run_one 带回的 args 是 None，
        # 上面那句会 AttributeError（原来就有，报的会是"NoneType has no attribute
        # rate"这种和真因无关的错）。
        "tcode": read_tcode(out, fps=(args.rate or args.fps) if args else None),
        "out_dir": str(out),
        "params": {k: v for k, v in vars(args).items() if k in _ALLOWED} if args else {},
        "axes": axes,
        "report": buf.getvalue(),
    }
    if err:
        result["error"] = err
    return result


def saved_person_timeline(slug: str) -> list[dict] | None:
    """读回**人物时间线**（哪一段用哪条轨迹），没有就 None。

    它由 `convert()` 在时段模式下写进产物目录（`person_timeline.json`），
    提交时跟着整个预览目录一起进正式目录。**故意不写进 funscript**：那是播放器
    读的脚本，多塞字段进去就是污染；这份是"这条脚本是怎么拼出来的"的账，
    单独一个文件，谁也不认它也没关系。

    读的顺序：**先正式产物、再预览草稿**。正式那份对应界面上正在显示的那套脚本
    （打开素材时取的就是正式产物），预览那份是"重算过但还没提交"的草稿——
    两个都在时以正式为准，否则会出现"脚本是 A、时间线画的是 B"。

    返回归一化成 `[{"from": int, "tid": int}]`，`to` 由下一段的 `from` 决定，
    和前端 personSegs 的形状一致。
    """
    for d in (paths.FUNSCRIPTS / slug, draft_dir(slug)):
        f = d / "person_timeline.json"
        if not f.exists():
            continue
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out = [{"from": int(x.get("from") or 0), "tid": int(x["tid"])}
               for x in raw if isinstance(x, dict) and x.get("tid") is not None]
        if out:
            out.sort(key=lambda x: x["from"])
            return out
    return None


def _read_axes(d: Path) -> dict[str, dict]:
    """把目录里的 funscript 读回来交给前端。"""
    out: dict[str, dict] = {}
    for axis, path in paths.funscript_files_in(d).items():
        try:
            out[axis] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return out


def _write_axes(out: Path, name: str, merged: dict[str, list[dict]]) -> list[Path]:
    """把「轴名 -> actions」写成一套产物：主轴单文件 + 各次轴 + multi 合体。

    `name` 决定文件名，而它来自 npz 所在目录名（to_funscript 里是
    `args.name or npz.parent.name`）——**换过轨迹或坐标系，名字就变了**。
    所以写完之后要把目录里旧名字那套清掉，否则同一份脚本会以两个名字并存，
    而 funscript_files_in 是按后缀取第一个的（见 `_wipe` 的说明）。

    **返回这次写出来的路径**，调用方拿它去清旧名字。别让调用方自己
    "先列一遍目录、写完再删那一列"——新旧名字相同时那等于把刚写的删掉
    （踩过：替换完一段，返回的轴全是空的）。
    """
    written: list[Path] = []
    for axis, acts in merged.items():
        suffix = paths.AXIS_SUFFIX[axis]
        fn = f"{name}.funscript" if suffix is None else f"{name}.{suffix}.funscript"
        p = out / fn
        p.write_text(json.dumps(
            {"version": "1.0", "inverted": False, "range": 100, "actions": acts},
            ensure_ascii=False), encoding="utf-8")
        written.append(p)
    p = out / f"{name}.multi.funscript"
    p.write_text(json.dumps({
        "version": "1.1", "inverted": False, "range": 100,
        "axes": [{"id": a, "actions": merged[a]} for a in paths.AXIS_SUFFIX if a in merged],
    }, ensure_ascii=False), encoding="utf-8")
    written.append(p)
    return written


def _splice_segment(out: Path, seg_dir: Path, want_frames: int | None = None) -> str | None:
    """把刚算好的这一段（seg_dir）替换进 out 里的整条脚本，段外一个字节都不动。

    成功返回 None，失败返回给用户看的说明——**失败时 out 必须原封不动**
    （调用方依赖这一点：算不出来就不该动已有脚本）。

    "哪一段"直接按 `at` 比：段的时间戳是绝对帧号（`--range` 负责），段自己的
    首尾 `at` 就是要替换的闭区间，不需要在这里重新推帧率。

    `want_frames` 是整条素材的帧数（`--rate` 重采样时传 None，那时产物帧数和
    素材帧数本来就对不上）。用它堵住"半截脚本当基准"的第二种形态：只查开头
    是不是第 0 帧，挡不住**从 0 开始但提前结束**的半截——那正是老版「单独重算」
    留下的样子，拿它替换的话，缺的尾巴会一直缺下去，而且看不出哪里不对。
    """
    base = paths.funscript_files_in(out)
    ref = _read_actions(base.get("L0"))
    short = not ref or ref[0]["at"] != 0 or (
        want_frames is not None and len(ref) < want_frames)
    if short:
        have = f"只有 {len(ref)} 帧" if ref else "读不出主轴"
        want = f"，整条是 {want_frames} 帧" if want_frames is not None else ""
        return (f"预览里那份脚本不是整条（{have}{want}）——单独重算是把**已有的**"
                "脚本换掉一段。先在「参数与重算」点一次重算，生成整条以后再来。")
    seg_files = paths.funscript_files_in(seg_dir)
    seg = {a: _read_actions(p) for a, p in seg_files.items()}
    if not any(seg.values()):
        return "这一段没有产出任何轴"
    # 名字取自这次真正写出来的文件（to_funscript 从 npz 目录名取名）——
    # 不在这里重写一遍取名规则，规则改了两边就会漂。
    l0 = seg_files.get("L0") or base.get("L0")
    name = l0.name[: -len(".funscript")] if l0 else "script"

    merged: dict[str, list[dict]] = {}
    for axis in paths.AXIS_SUFFIX:
        b = _read_actions(base.get(axis))
        s = seg.get(axis) or []
        if not b:
            acts = s                                   # 基底没有这条轴：只有这一段
        elif not s:
            acts = b                                   # 这一段没产出这条轴：原样保留
        else:
            t0, t1 = s[0]["at"], s[-1]["at"]
            acts = [a for a in b if a["at"] < t0] + s + [a for a in b if a["at"] > t1]
        if acts:
            merged[axis] = acts

    # tcode 的行号**就是绝对帧号**（一帧一行），所以"替换哪几行"= "替换哪几帧"，
    # 行号从基准的 L0 上数出来即可，不用重新推帧率。
    lines = _tcode_lines(out)
    seg_lines = _tcode_lines(seg_dir)
    tc_lines: list[str] = []
    if lines and seg_lines and len(lines) == len(ref) and seg.get("L0"):
        t0, t1 = seg["L0"][0]["at"], seg["L0"][-1]["at"]
        i0 = sum(1 for a in ref if a["at"] < t0)
        i1 = sum(1 for a in ref if a["at"] <= t1)
        if 0 <= i0 <= i1 <= len(lines):
            tc_lines = lines[:i0] + seg_lines + lines[i1:]
    elif seg_lines:
        tc_lines = seg_lines                           # 基准就没有 tcode，只有这一次的
    # 对不上（行数和帧数不一致，说明文件被人动过）就整份不写：宁可没有 tcode，
    # 也不要一份和 funscript 各说各话的——播放器读 funscript、模拟器读 tcode，
    # 两者对"有哪些帧"的说法必须一致。

    # 到这一步才动 out。写出来的那份是**新名字**，删的是**没被重写过的旧文件**
    # （名字相同的不算旧文件——照名字判，不照时间判）。
    written = _write_axes(out, name, merged)
    if tc_lines:
        p = out / f"{name}.tcode.txt"
        p.write_text("\n".join(tc_lines) + "\n", encoding="utf-8")
        written.append(p)
    keep = {p.name for p in written}
    for p in out.iterdir():
        # 只清**脚本文件**里没被重写的那份（换了轨迹/坐标系之后名字会变）。
        # 别的文件一概不动：预览目录里还躺着 person_timeline.json（人物时间线的
        # 记录），它不是"这一次要改的那一段"。
        if p.is_file() and p.name not in keep and (
                p.name.endswith(".funscript") or p.name.endswith(".tcode.txt")):
            p.unlink()
    return None


def _merge_segments(out: Path, seg_dirs: list[Path], name: str) -> None:
    """把各段的产物按时间拼成一份。

    段的产物**时间戳是绝对帧号**（`--range` 写的），所以这里只要按 `at` 排序；
    段之间不重叠，实际上就是顺序相接。
    """
    merged: dict[str, list[dict]] = {}
    for axis in paths.AXIS_SUFFIX:
        acts: list[dict] = []
        for d in seg_dirs:
            acts += _read_actions(paths.funscript_files_in(d).get(axis))
        if acts:
            acts.sort(key=lambda x: x["at"])
            merged[axis] = acts
    if merged:
        _write_axes(out, name, merged)
    # tcode 直接按段落顺序接起来（每行是绝对轴值，不含时间戳，顺序即时间）
    lines: list[str] = []
    for d in seg_dirs:
        lines += _tcode_lines(d)
    if lines:
        (out / f"{name}.tcode.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    # 段目录只是中间产物，别留在预览目录里让用户以为那是正式脚本
    for d in seg_dirs:
        shutil.rmtree(d, ignore_errors=True)


def clear(slug: str | None = None) -> None:
    """清缓存。缓存键是 (slug, space)，所以按 slug 清要清掉它的所有坐标系。"""
    with _LOCK:
        if slug is None:
            _SEQ.clear()
        else:
            for k in [k for k in _SEQ if k[0] == slug]:
                del _SEQ[k]
