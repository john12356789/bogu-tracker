"""批量跑管线。六个步骤可任意组合，默认**不跑第 6 步（渲染视频）**。

步骤之间有依赖（见下面的 STEP_DEPS）。命令行下任意组合都放行——缺前置的
后果是报错或走慢路，跑之前说清楚就够了；Web 那条路会按依赖置灰并挡住提交。

步骤
----
  1 preprocess  预处理：降采样到 1280p、>40fps 的源统一到 30fps
  2 boxes       检测框 + HMR2 特征（跳过 ViTPose）
  3 rtmpose     RTMPose-x 384x288 关键点（实测胯部重投影误差比 ViTPose-H 低 28%）
  4 gvhmr       推理
  5 export      导出 npz
  6 render      四格视频：原片 | 叠加 | mesh正面 | mesh侧面

第 6 步默认不跑——它只产出给人看的预览，占的时间却不少。
要看视频随时用 --steps ...,render 补跑，前五步的产物都在，不用重算。

给 Web 后端用的进度协议
----------------------
每行以 `@@EVENT ` 开头，后面是 JSON。为什么不让调用方自己去解析子工具的
输出：tqdm 的格式、渲染的 `渲染 n/N` 格式、以及以后可能的变化，都只有这里
知道。对外只暴露一个稳定的协议。

  {"kind":"step","index":1,"total":5,"id":"preprocess","name":"预处理","state":"start"}
  {"kind":"progress","index":1,"frac":0.42}          # 只有能拿到子进度的步骤才有
  {"kind":"step",...,"state":"done","elapsed":12.3}
  {"kind":"failed","index":4,"message":"..."}

**哪些步骤没有子进度**：第 4 步做不到——GVHMR 的 predict() 是对整段序列的
单次 forward，没有分块也没有回调钩子（见 hmr4d/model/gvhmr/gvhmr_pl_demo.py:38）。
要加就得改上游仓库，而本项目刻意不碰上游文件。第 1/3/5 步目前静默。

用法
----
    python src/batch_run.py                          # 全部视频，跑前 5 步
    python src/batch_run.py --steps gvhmr,export     # 只跑指定步骤
    python src/batch_run.py --steps preprocess,boxes,rtmpose,gvhmr,export,render
    python src/batch_run.py --only 橘小狗 --force
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import tracklet_merge
from mesh_npz import mesh_path_for
from preprocess import video_readable  # 纯标准库；同目录，run 时 sys.path[0] 就是 src/


# 开头的 ASCII 段短于这个长度时，它就不算"一条素材的身份"了——一个 1~2 字符的
# 段只是名字第一个字符的巧合（"1分40秒-…" 和 "1月新番-…" 都会取到 "1"），
# 两条素材会共用同一个产物目录。见 ascii_slug。
MIN_SLUG = 3


def ascii_slug(name: str) -> str:
    """把视频名压成 ASCII 短名。**这就是一条素材在盘上的身份**。

    GVHMR 用 hydra 解析命令行覆盖项（video_name=...、output_root=...），
    而 hydra 的覆盖语法**不接受非 ASCII**——中文名会让它直接抛
    LexerNoViableAltException。B 站视频开头的 BV 号本身是 ASCII，取它即可。

    注意连字符也在 ASCII 段里，所以 "BV1KJes6KEhJ-Lo老师-…" 取到的是
    "BV1KJes6KEhJ-Lo"（盘上的目录就是这个，不是 BV1KJes6KEhJ）。

    **兜底值必须唯一**。原来这里是 `return slug or "video"`，而 "video" 不是任何
    一条素材的身份：名字开头不是 ASCII 的（中文标题）全都映射到同一个 "video"，于是
    outputs/video/ 和 processed/video_1280p_30fps.mp4 被所有这类素材共用，
    后跑的把先跑的整个顶掉。实测 workspace/videos/旗袍—卡点爆摇.mp4 就是这样。

    改用名字的短哈希：同一个名字永远同一个 slug（产物找得回来）、不同名字
    几乎必不同（不会互相覆盖）、纯 ASCII（hydra 和 SAFE_SLUG 都过）。
    中文标题的素材没有 ASCII 身份可言，哈希是这里唯一诚实的选择——
    从名字里"翻译"出一个英文名只会引入第二套命名规则。
    """
    m = re.match(r"^[A-Za-z0-9_\-]+", name)
    slug = (m.group(0) if m else "").strip("-_")
    if len(slug) >= MIN_SLUG:
        return slug
    return "v_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
VIDEOS = ROOT / "workspace" / "videos"
PROCESSED = ROOT / "workspace" / "processed"
OUTPUTS = ROOT / "workspace" / "outputs"

# 两套环境：GVHMR 在 3.10，rtmlib / rerun / cv2 在 3.12
#
# PY_GVHMR **可以用环境变量覆盖**（BOGU_PY_PIPELINE）。写死的那条是本机的默认值，
# 只有打包成桌面应用时才不得不改：那台机器上这个环境装在哪只有用户知道，而
# 桌面壳的"环境检查"里正好让他选过一次——选完写进配置，启动后端时带过来。
# 没设这个变量时行为与以前逐字节相同，开发时的命令行不受影响。
PY_GVHMR = Path(os.environ.get("BOGU_PY_PIPELINE") or "E:/anaconda3/envs/gvhmr/python.exe")
PY_MAIN = Path(sys.executable)

# 规范顺序。执行时按这个顺序过滤用户选中的步骤，不按用户给的顺序——
# 后面的步骤依赖前面的产物，顺序错了只会得到一堆难懂的报错。
STEP_ORDER = ["preprocess", "boxes", "rtmpose", "gvhmr", "export",
              "merge", "tracks", "render"]
STEP_NAME = {
    "preprocess": "预处理",
    "boxes": "检测框 + HMR2 特征",
    "rtmpose": "RTMPose 2D 关键点",
    "gvhmr": "GVHMR 推理",
    "export": "导出 npz",
    "merge": "归并人物",
    "tracks": "多人轨迹",
    "render": "渲染 quad",
}
# 界面上不列出来的步骤（仍然能跑：作业队列按步骤名拼命令行，命令行也能 --steps）。
#
# 「归并人物」和「多人轨迹」属于这一种：它们的入口都在**「人物」面板**。
# 这两步是同一件事的两半——"把画面里的人各跑一份产物"，而"谁是谁"这件事只有
# 人物面板说得清（哪几段是同一个人、哪几分是同一个人）。摆在左边的勾选面板里，
# 用户就得在两个地方来回跳：先在面板里合人，再回左边勾第 6 步，反着来还会拿到
# 一句"没有分组决策，什么都不做"（真发生过）。
#   merge   把确认过的几段合成一条，再重建那一条
#   tracks  其余的每个人各重建一份
# 面板里那两个按钮提交的就是 `steps: ["merge"]` / `["tracks"]`。
HIDDEN_STEPS = frozenset({"merge", "tracks"})

# 前面这几步是默认跑的；**后面三个不默认**：
#   render  只是给命令行补跑正式 quad.mp4 用的
#   merge   （把同一个人的几段轨迹合成一条）要花一次全片重建，而且**只有在
#            「人物」面板里确认过分组之后才有事可做**——没决策时它是空步。
#            放在 tracks 前面：先把人并好，tracks 就不会再去跑那几段碎片。
#   tracks  （其余轨迹各跑一遍）代价是乘性的，必须手动开——见 _run_extra_tracks
#
# 「归并」在算法和默认值上是同一个决定：**它整套都不自动发生**（见
# tracklet_merge.effective 的文件头那段——并错人比多看几行碎片严重）。
DEFAULT_STEPS = [s for s in STEP_ORDER if s not in ("merge", "tracks", "render")]

# 各步骤的可调参数（步骤 id -> 参数表）。
#
# **这里是唯一的定义处**：界面拿它渲染输入框/选项（中文名、默认值、范围、说明、
# 甚至命令行长什么样），后端拿它做校验和默认值。两边各写一份的话，界面能把值
# 调成后端不认的，而错误要等到跑完几十分钟才在日志里出现。
#
# `flag` 是给作业队列拼命令行用的——`jobs.py` 只按这张表拼，不自己写一遍
# "哪些步骤有哪些参数"，那种映射抄第二遍必定漂。
STEP_OPTS: dict[str, list[dict]] = {
    # 「导出 npz」。默认只导相机系：它是整条链路都在用的那份（算脚本的默认来源、
    # 渲染的主输入、'这条素材导出了没有'的判据），而且带相机内外参、诊断全可用。
    # 世界系是**另一套独立重建**（不是刚性变换），只有 3D 视图切世界系、以及渲染
    # 的世界系格子要用。两份各约 0.1 MB/帧 —— 6727 帧的素材上两份合计 1.5 GB，
    # 只导相机系直接省掉一半。
    "export": [
        {"key": "spaces", "label": "导出坐标系", "kind": "choice",
         "flag": "--export-spaces",
         "default": "incam",
         "choices": [
             {"value": "incam", "label": "相机系"},
             {"value": "incam,global", "label": "两个都要"},
             {"value": "global", "label": "世界系"},
         ],
         "hint": "相机系带相机内外参，算脚本和渲染都用它；世界系只能给 3D 视图的"
                 "世界系模式和渲染的世界系格子用。每份约 0.1 MB × 素材帧数"
                 "（3000 帧 ≈ 300 MB，6700 帧 ≈ 760 MB）。先只导相机系，"
                 "以后要看世界系再补一次就行。"},
    ],
    # 「多人轨迹」。为什么需要那两个闸（实测 4K60 的 MV：6727 帧里有 110 个切点）：
    # 切一次镜头 ByteTrack 就丢一次人，跟出的"轨迹"全是碎片——21 条 ≥3% 的，
    # 最高的才 6%，另有 426 条碎片被当噪声丢掉。**每条轨迹都要走一遍完整流水线**，
    # 而 prep_boxes 会把 bbx 补成全长再对**全部帧**提 HMR2 特征，于是每条 ≈20 分钟、
    # ≈1.8 GB。20 条就是六七个小时、几十 GB，而产物是"在两次不相关镜头之间线性
    # 插值的框"，没有意义。
    "tracks": [
        {"key": "min_coverage", "label": "占比阈值", "kind": "percent",
         "flag": "--track-min-coverage",
         "default": 0.20, "min": 0.02, "max": 1.0, "step": 0.01,
         "hint": "只跑占比不低于这个数的轨迹。占比 3% = 只在 3% 的帧里跟住了人，"
                 "这种多半是切镜头切出来的碎片"},
        {"key": "max_tracks", "label": "最多条数", "kind": "int",
         "flag": "--track-max-tracks",
         "default": 3, "min": 1, "max": 10, "step": 1,
         "hint": "按累计面积取前 N 条。这是花销的上限：每条 ≈20 分钟、≈1.8 GB"},
        # 界面上**不显示**的一项：它是「人物」面板每一行那个「重建」按钮用的，
        # 只出现在那一次作业的参数里。给了就只跑这几条，阈值和条数上限都不看
        # ——用户在一行上点了"重建"，那条轨迹就该跑，不该被阈值再筛一遍。
        {"key": "only", "label": "只跑这些轨迹", "kind": "text",
         "flag": "--track-only", "default": "", "hidden": True,
         "hint": "逗号分隔的轨迹号。面板上每一行的「重建」按钮带的就是这个。"},
    ],
}


def step_opts_default(step: str) -> dict:
    return {o["key"]: o["default"] for o in STEP_OPTS.get(step, [])}


def normalize_step_opts(step: str, raw: dict | None) -> dict:
    """把界面/命令行给的参数夹到合法取值。缺的、不认识的都用默认值。"""
    out = step_opts_default(step)
    for o in STEP_OPTS.get(step, []):
        v = (raw or {}).get(o["key"])
        if v is None or isinstance(v, bool):
            continue
        if o["kind"] == "choice":
            # 选择项只能取表里列出的值。**不做"猜一个相近的"**：拼错的值落到
            # 默认值上比落到别的坐标系上安全得多。
            if v in [c["value"] for c in o["choices"]]:
                out[o["key"]] = v
            continue
        if o["kind"] == "text":
            # 文本项**不夹数值范围**，只做长度和字符限制：它可能来自 URL/body。
            out[o["key"]] = "".join(str(v).split())[:200]
            continue
        try:
            v = int(v) if o["kind"] == "int" else float(v)
        except (TypeError, ValueError):
            continue
        out[o["key"]] = min(max(v, o["min"]), o["max"])
    return out


def normalize_all_step_opts(raw: dict | None) -> dict[str, dict]:
    """整包规范化：{步骤 id: 参数}。只保留**这一步真有参数**的项。"""
    raw = raw or {}
    return {s: normalize_step_opts(s, raw.get(s)) for s in STEP_OPTS}

# 每一步的前置（只写**直接**依赖，传递闭包由 prereqs() 算）。
#
# 两种"缺了就出事"都记在这里，都当硬前置：
#   · 直接失败：除 preprocess 外的每一步都要预处理产物（process_one 里运行时
#     也会挡一道）；export 没有 hmr4d_results.pt 就报文件不存在；render 同理。
#   · 走错路：gvhmr 少了 boxes 会退回 GVHMR 自带的 YOLO 自己算一遍，少了
#     rtmpose 就换成它的 ViTPose——能跑完，但 2D 关键点不再是 export 要读的
#     那份 vitpose.pt。白等一轮，还换了实现。
#     （这条退路现在**已经不可能走**：run_gvhmr 缺产物直接拒绝跑，ViTPose-H 的
#     权重也已从盘上删掉；这里保留这段是因为它正是"当硬前置"的理由。）
#
# 界面（/api/steps）拿这张表把后面几步置灰；后端在 POST /api/jobs 上再挡一次。
# 两边都不能省：界面不是唯一入口，curl 一样能提交作业。
STEP_DEPS: dict[str, list[str]] = {
    "preprocess": [],
    "boxes": ["preprocess"],
    "rtmpose": ["preprocess", "boxes"],
    "gvhmr": ["boxes", "rtmpose"],
    "export": ["gvhmr"],
    # 归并：只读第 2 步的 bbx_all.json（分组是它算的）+ 人工确认过的
    # people_merge.json，然后把每个合并组当一条轨迹跑一遍 2~5 步。
    # 前置挂 boxes 就够——它不需要主轨迹的 npz。
    "merge": ["boxes"],
    # 其余轨迹各跑一遍。**前置直接挂 export**：它每一条都要把 3~5 步走完，
    # 没有主轨迹那份 export 出来的 npz 就没法比、也没法确认主轨迹是好的。
    "tracks": ["export"],
    "render": ["export"],
}


def prereqs(step: str) -> list[str]:
    """step 的**全部**前置（传递闭包），按 STEP_ORDER 排。

    界面要问的是"这一步之前还差哪些"，不是只有直接上一层——只给直接依赖的话，
    置灰时只能报"缺 RTMPose"，而 RTMPose 自己又缺检测框，用户得一轮一轮试出来。
    """
    seen: set[str] = set()
    stack = list(STEP_DEPS[step])
    while stack:
        d = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        stack.extend(STEP_DEPS[d])
    return [s for s in STEP_ORDER if s in seen]


PREREQS: dict[str, list[str]] = {s: prereqs(s) for s in STEP_ORDER}


def check_prereqs(steps: list[str], products: dict[str, bool]) -> str | None:
    """这套步骤能不能跑。返回 None = 能，否则是一句给用户看的说明。

    判据是"**勾了 或 盘上已有产物**"，不是"前置必须被勾上"：只勾后面几步是
    正常用法——workspace 里的素材绝大多数已经算过，补个导出不该被拦住。

    **只在 Web 这条路上用**（jobs.validate）。命令行刻意不做这个检查：
    CLI 是按文件名批量跑的，各条素材的产物齐不齐本来就不一样，一刀切会拦掉
    正常批处理。（命令行那条入口由 run_gvhmr 自己在跑之前挡：缺 bbx/vitpose/
    vit_features 就直接拒绝，而不是让上游悄悄换成 ViTPose 再跑一遍。）
    """
    have = set(steps) | {s for s, ok in products.items() if ok}
    problems = []
    for s in steps:
        miss = [STEP_NAME[p] for p in PREREQS[s] if p not in have]
        if miss:
            problems.append(f"「{STEP_NAME[s]}」缺 " + "、".join(miss))
    if not problems:
        return None
    return "；".join(problems) + "。把它们也勾上，或先跑一次让盘上有产物。"


EVENT_PREFIX = "@@EVENT "


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(kind: str, **kw) -> None:
    """给调用方（Web 后端）的结构化事件。人看的日志走 log()。"""
    print(EVENT_PREFIX + json.dumps({"kind": kind, **kw}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# 子进程执行 + 进度解析
# ---------------------------------------------------------------------------
# 形如 "1234/2745" 的完成度。要求分母 >= 10，否则会把 "[1/6]" 这种
# 小数字误判成进度。
_FRAC_RE = re.compile(r"(\d+)\s*/\s*(\d+)(?!\d)")


def parse_fraction(line: str) -> float | None:
    """从一行输出里取完成度。取最后一个可信的 n/m。

    兼容两种已知格式：
      tqdm      "HMR2 Feature:  45%|####  | 1234/2745 [00:12<00:15, 98.7it/s]"
      render    "  渲染 125/2444"
    """
    best = None
    for m in _FRAC_RE.finditer(line):
        num, den = int(m.group(1)), int(m.group(2))
        if den >= 10 and num <= den:
            best = num / den
    return best


def stream_proc(cmd: list[str], cwd: Path, on_line) -> tuple[int, list[str]]:
    """跑子进程并**逐行流式**读出输出（stderr 并进 stdout）。

    三个都不能省的细节：

    1. 不能用 subprocess.run(capture_output=True)：子进程的输出要等它整个跑完
       才拿得到，进度只能事后补，等于没有。
    2. tqdm 默认写 stderr，所以必须把 stderr 合并进来。
    3. **必须用 os.read，不能用 file.read(n)**。管道上的 `read(n)` 会阻塞到凑满
       n 字节或 EOF 才返回，输出会全积在最后一次性到达——这正是本项目踩过的坑：
       渲染用 `print(..., end="\\r")` 刷新进度，结果十几个进度点一个都没透出来。
       os.read 一次系统调用，有多少返多少。同理要按 \\n 和 \\r 两种边界切行，
       否则 tqdm 和渲染的进度行永远凑不成一行。
    """
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0,                                    # 二进制裸管道，不做缓冲
    )
    tail: deque[str] = deque(maxlen=15)
    dec = codecs.getincrementaldecoder("utf-8")("replace")   # 多字节字符可能跨块
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    buf = ""

    def feed(text: str) -> None:
        nonlocal buf
        buf += text
        while True:
            idxs = [i for i in (buf.find("\n"), buf.find("\r")) if i >= 0]
            if not idxs:
                break
            cut = min(idxs)
            line, buf = buf[:cut], buf[cut + 1:]
            if line.strip():
                tail.append(line.rstrip())
                on_line(line)

    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        feed(dec.decode(chunk))
    feed(dec.decode(b"", final=True))
    if buf.strip():
        tail.append(buf.rstrip())
        on_line(buf)

    proc.wait()
    return proc.returncode, list(tail)


# 原生崩溃的退出码（Windows）。**不是 Python 的异常**：进程是被系统/驱动/硬件
# 直接干掉的，所以没有 traceback、也没有正常的错误信息，只剩一个数。
# 单列出来是为了两件事：给用户一句能懂的话，以及在**这一种失败上重试一次**。
#   0xC0000409 栈缓冲溢出（也用于"被系统强杀"：提交内存耗尽时的快速失败）
#   0xC0000005 访问违例   0xC000001D 非法指令   0xC0000374 堆损坏
#   0xC0000602 fail-fast（__fastfail，比 0409 更明确的"立即终止"）
#   0xC000013A Ctrl-C 终止（用户按了取消）
NATIVE_CRASH = {
    0xC0000409: "进程被系统强制终止（栈缓冲溢出 / 提交内存耗尽时的快速失败）",
    0xC0000005: "访问违例（多半是显卡驱动或某个原生库崩了）",
    0xC000001D: "非法指令（CPU 不支持该指令，或二进制不匹配）",
    0xC0000374: "堆损坏（原生库的内存被写坏）",
    0xC0000602: "进程被立即终止（fail-fast）",
    0xC000013A: "被 Ctrl-C 终止",
}


def explain_exit(code: int) -> str:
    """退出码 -> 人话。原生崩溃的码是**无符号 32 位**，Python 给的是有符号的负数，
    所以要 & 0xFFFFFFFF 之后再查。查不到就原样报数。"""
    text = NATIVE_CRASH.get(code & 0xFFFFFFFF)
    return f"{text}（0x{code & 0xFFFFFFFF:08X}）" if text else f"退出码 {code}"


def run_step(cmd: list[str], desc: str, index: int, on_prog,
             cwd: Path = ROOT, attempts: int = 2) -> None:
    log(f"    $ {desc}")

    def on_line(line: str) -> None:
        log(f"      {line.rstrip()}")
        frac = parse_fraction(line)
        if frac is not None:
            on_prog(frac)

    for attempt in range(1, max(1, attempts) + 1):
        code, tail = stream_proc(cmd, cwd, on_line)
        if code == 0:
            return
        crash = code & 0xFFFFFFFF in NATIVE_CRASH
        # **只在原生崩溃上重试**：那是环境性的（显存/内存被占、驱动抽风），
        # 重来一次常常就好。干净的退出码 1 是逻辑错误，重试只会再错一遍，
        # 白等几分钟还掩盖了真正的问题。
        #
        # 重试便宜：上一步的产物都在盘上，各步的"已有就跳过"会让它直接回到
        # 崩掉的那一步（实测第 4 步崩掉时，前 3 步的产物都还在）。
        if crash and attempt < attempts:
            log(f"    ⚠ {desc} 原生崩溃：{explain_exit(code)}——环境问题（显存/内存"
                f"被占或驱动抽风），10 秒后重试一次")
            time.sleep(10)
            continue
        raise RuntimeError(f"{desc} 失败（{explain_exit(code)}）:\n" + "\n".join(tail))


def probe_fps(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    num, den = (out.split("/") + ["1"])[:2]
    return float(num) / float(den or 1)


# ---------------------------------------------------------------------------
# 各步骤
# ---------------------------------------------------------------------------
def step_preprocess(raw: Path, name: str, ctx: dict, on_prog) -> None:
    # 降采样是 ffmpeg 重编码，不会自己判断该不该做——不像后面几步有 "已有就跳过"
    # 的逻辑。这里补一个，否则在已处理过的素材上点一次"跑"就会白编一遍 1280p。
    #
    # **但"在不在"不等于"能不能用"**：编码写到一半被打断会留下大小正常的残件
    # （MP4 的索引写在文件末尾），而它会被这个跳过逻辑当成做完了。判据要加一道
    # 能不能读，读不了就重编——ffmpeg 那边是 `-y`，直接覆盖。
    if not ctx["force"]:
        existing = sorted(PROCESSED.glob(f"{name}_*p*.mp4"))
        good = [p for p in existing if video_readable(p)]
        if good:
            log(f"    已有预处理产物，跳过：{good[-1].name}")
            return
        for p in existing:
            log(f"    ⚠ 预处理产物读不了（多半是上次编码被打断留下的残件），重编：{p.name}")
    fps = probe_fps(raw)
    want_fps = 30.0 if fps > 40 else None
    PROCESSED.mkdir(parents=True, exist_ok=True)
    log(f"    原始 {fps:.0f}fps -> ASCII 名 {name}")
    run_step([str(PY_MAIN), str(SRC / "preprocess.py"), str(raw),
              "-o", str(PROCESSED), "--name", name, "--height", "1280"]
             + (["--fps", "30"] if want_fps else []),
             "降采样", 1, on_prog)


def step_boxes(raw: Path, name: str, ctx: dict, on_prog) -> None:
    run_step([str(PY_GVHMR), str(SRC / "prep_boxes.py"),
              "--video", str(ctx["proc_video"]), "--output-root", str(ctx["rt_root"])],
             "YOLO + HMR2", 2, on_prog)


def _run_extra_tracks(ctx: dict, on_prog) -> None:
    """多人素材：为**其余**每条有效轨迹各跑一遍 3~5 步。

    **只有「多人轨迹」这一步被勾上时才会走到这里**（它不在 DEFAULT_STEPS 里）。
    原因是花销是乘性的：每条轨迹都要把整条流水线走一遍，而 prep_boxes 会把 bbx
    补成全长、对全部帧提 HMR2 特征——实测 6727 帧的 4K60 素材上每条 ≈20 分钟、
    ≈1.8 GB。一条切镜头很碎的素材能跟出二十几条，那就是六七个小时、几十 GB，
    而每条只覆盖百分之几、产物是跨镜头插值出来的框。所以默认不跑，且有两个闸：

      占比阈值 min_coverage  低于它的不当"另一个人"看（切镜头切出来的碎片）
      最多条数 max_tracks    按累计面积取前 N，给花销封顶

    主轨迹仍然写 `gvhmr_rtmpose/`（一个字都没动），其余写
    `gvhmr_rtmpose__t<tid>/` —— 所以**已经跑过的单人素材零迁移、零影响**。

    每一步都有"已有就跳过"，所以重跑一条已经跑完的轨迹几乎不花时间。
    """
    allp = ctx["rt_dir"] / "bbx_all.json"
    if not allp.exists():
        return
    try:
        info = json.loads(allp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    segs = info.get("segments", [])
    if len(segs) <= 1:
        log("    （只跟出一条轨迹，没有其余的人要跑）")
        return
    # 其余轨迹一律走完 2~5 步。**不再看用户还勾了哪几步**：这一步本身就是手动
    # 勾出来的（它不在 DEFAULT_STEPS 里），勾了就说明要它跑；而每一步都有"已有
    # 就跳过"，所以已经跑过的轨迹几乎不花时间。
    #
    # （原来这里有个"只勾了前两步就不跑推理"的闸。那是这一步还挂在流程末尾自动
    # 执行时的遗留——那时用户完全可能只想要检测框。现在它只能被显式勾选。)
    want = ["rtmpose", "gvhmr", "export"]

    # 从 ctx 里读出来的东西**自己再规范化一遍**，不假设调用方已经做过：少一个键
    # 就是 KeyError，而那会在"跑到一半"的时候炸，看起来像"这一步坏了"。
    opts = normalize_step_opts("tracks", (ctx.get("step_opts") or {}).get("tracks"))
    min_cov, max_n = opts["min_coverage"], opts["max_tracks"]
    others = segs[1:]
    # 「归并人物」已经把它们合成一条了：再按碎片各跑一遍是白跑，而且跑出来的
    # 是同一批帧、更差的框（碎片各自缺的那些帧靠插值补）。
    #
    # **必须在按阈值截断之前剔掉**：碎片通常占比高（一条 40% 的碎片会稳稳占住
    # 前几名），先截断再剔的话，名额被这几条已经合并掉的碎片吃光，剩下那些真正
    # 独立的人一条都轮不到——现象就是"归并之后点『重建其余人物』什么都没跑"。
    absorbed = {t for g in _merged_groups(ctx) for t in g["tids"]}
    pool = [s for s in others if s["tid"] not in absorbed]
    if len(pool) < len(others):
        log(f"    其中 {len(others) - len(pool)} 条已经并进「归并人物」的组里，"
            f"不在这里单独跑")
    # 「人物」面板每一行那个「重建」按钮：指定轨迹号就**只跑它**，阈值和条数
    # 上限都不看——用户在一行上点的那一下就是明确要求，不该被阈值再筛掉一遍。
    only = [int(x) for x in str(opts.get("only") or "").replace("，", ",").split(",")
            if x.strip().isdigit()]
    if only:
        keep = [s for s in pool if s["tid"] in only]
        miss = [t for t in only if t not in {s["tid"] for s in pool}]
        if miss:
            log(f"    指定的 {miss} 不在可跑的清单里（不存在，或者已经并进某个人的组里）")
    else:
        keep = [s for s in pool if s["coverage"] >= min_cov][:max_n]
    best = max(s["coverage"] for s in others)
    main_cov = segs[0]["coverage"]
    if not keep:
        if only:
            # 指定了轨迹却一条都跑不了：原因上面那行已经说了（不存在/已并进组）。
            return
        # 这条是 BV1A 那种素材的常态：主轨迹自己都只占几个百分点，说明**跟本
        # 没跟住人**，其余就更不用看了。说清楚"为什么一条都没跑"，否则界面
        # 上只有主轨迹的产物，用户会以为是功能坏了。
        # （另一种"一条都没跑"是全都并进了某个组——那上面那行已经说过了。）
        if pool:
            log(f"    剩下 {len(pool)} 条都在 {min_cov:.0%} 阈值以下（最高 "
                f"{max(s['coverage'] for s in pool):.1%}，主轨迹自己才 {main_cov:.1%}）"
                f"——多半是切镜头太碎、跟踪反复丢失。要放宽：调「人物」面板里的"
                f"占比阈值（或 --track-min-coverage）")
        return
    if len(keep) < len(pool):
        log(f"    剩下 {len(pool)} 条里只跑 {len(keep)} 条："
            f"{len(pool) - len(keep)} 条低于 {min_cov:.0%} 或超出 {max_n} 条上限"
            f"（最高 {best:.1%}）")

    proc = ctx["proc_video"]
    log(f"    检测到 {len(segs)} 条有效轨迹（主轨迹 {main_cov:.1%}）："
        f"主轨迹外再跑 {len(keep)} 条，各自独立一份产物")
    for k, seg in enumerate(keep, start=1):
        tid = seg["tid"]
        troot = ctx["out_root"] / f"gvhmr_rtmpose__t{tid}"
        tdst = troot / proc.stem
        far = f"人物{k}·track{tid}"
        log(f"    ---- {far}：占 {seg['coverage']:.1%}，{len(seg['runs'])} 段 ----")
        # 检测框总要备（后面每一步都依赖它），且复用已跟好的轨迹段、不重跑 YOLO
        run_step([str(PY_GVHMR), str(SRC / "prep_boxes.py"),
                  "--video", str(proc), "--output-root", str(troot),
                  "--track", str(tid), "--tracks-from", str(allp)],
                 f"{far} 检测框", 2, on_prog)
        if "rtmpose" in want:
            run_step([str(PY_MAIN), str(SRC / "run_rtmpose.py"),
                      "--src", str(tdst), "--dst", str(troot), "--video", str(proc)],
                     f"{far} RTMPose", 3, on_prog)
        if "gvhmr" in want:
            run_step([str(PY_GVHMR), str(SRC / "run_gvhmr.py"),
                      "--video", str(proc), "-s", "--output_root", str(troot)],
                     f"{far} 推理", 4, on_prog)
        if "export" in want:
            # 其余轨迹**跟着同一次作业的导出设置走**（同一个作业里不该出现
            # "主轨迹只导相机系、别人导了两份"）。
            exp = normalize_step_opts("export", (ctx.get("step_opts") or {}).get("export"))
            run_step([str(PY_GVHMR), str(SRC / "export_gvhmr.py"),
                      str(tdst / "hmr4d_results.pt"),
                      "--spaces", str(exp["spaces"])],
                     f"{far} 导出 npz", 5, on_prog)
    ctx["extra_tracks"] = [s["tid"] for s in keep]


def _merged_groups(ctx: dict) -> list[dict]:
    """这条素材当前**生效**的分组——归并默认关闭（见 tracklet_merge.effective）。

    只有人在「人物」面板上点过"用这组"才会有分组；没点过就是空步。
    这里不再读 bbx_all.json 算建议：建议是界面上的东西，不是流水线的输入。
    """
    return tracklet_merge.effective(
        tracklet_merge.load(ctx["out_root"] / "people_merge.json"))


def _wipe_track_products(tdst: Path) -> None:
    """清掉一根里 2~5 步的产物。**不碰输入视频和 bbx_all.json**。

    合并清单变了就得清：框换了，特征 / 关键点 / 重建全是按旧框算出来的，而每一步
    都是"文件在就跳过"——不清的话跑完还是旧动作，界面上却显示"跑完了"。
    """
    for p in (tdst / "preprocess" / "bbx.pt", tdst / "preprocess" / "vit_features.pt",
              tdst / "preprocess" / "vitpose.pt"):
        p.unlink(missing_ok=True)
    for pat in ("hmr4d_results*.pt", "hmr4d_results*.npz", "*.mesh.npz"):
        for p in tdst.glob(pat):
            p.unlink(missing_ok=True)


def _run_merged_groups(ctx: dict, on_prog) -> None:
    """每个合并组当一条轨迹跑一遍 2~5 步（= 一个人一份产物）。

    **合并在重建之前**：把几段的框并成一条稠密框轨迹再重建，接缝从根上不存在
    （而不是各自重建完再把信号拼起来——那样每条各自定世界系和尺度，接缝处一定
    有台阶）。所以这里不是"把几份 npz 合起来"，是**重新跑一次**。

    产物落在组内最小 tid 的根里（`gvhmr_rtmpose__t<gid>`），顶掉那条碎片原来的
    产物——它已经被这一组包含了，留着只会占盘。
    """
    groups = _merged_groups(ctx)
    if not groups:
        log("    没有分组决策（在「人物」面板确认一次才有）——这一步什么都不做")
        return
    allp = ctx["rt_dir"] / "bbx_all.json"
    if not allp.exists():
        log(f"    找不到 {allp.name}（第 2 步的产物），跳过")
        return
    try:
        info = json.loads(allp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log("    bbx_all.json 读不了，跳过")
        return
    known = {s["tid"] for s in info.get("segments", [])}
    proc = ctx["proc_video"]
    want = ["rtmpose", "gvhmr", "export"]
    exp = normalize_step_opts("export", (ctx.get("step_opts") or {}).get("export"))

    for i, g in enumerate(groups, start=1):
        tids = sorted(t for t in g["tids"] if t in known)
        if len(tids) < 2:
            log(f"    ---- 跳过 {g['tids']}：这条素材的轨迹清单里没有它们 ----")
            continue
        gid = tids[0]                                  # 组内最小 tid = 这条"人"的编号
        troot = ctx["out_root"] / f"gvhmr_rtmpose__t{gid}"
        tdst = troot / proc.stem
        far = f"人{i}·归并{gid}"
        log(f"    ---- {far}：由 {len(tids)} 段并成（{' + '.join(map(str, tids))}）----")

        # 分组变了 → 旧产物是按别的框算的 → 先清掉。用一份小 JSON 当凭证，
        # 不读 bbx.pt（那是 torch 的 pickle，而 batch_run 这个解释器未必有 torch）。
        marker = troot / "merge_spec.json"
        old = None
        if marker.exists():
            try:
                old = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = None
        if old is not None and old != tids:
            log(f"    分组变了（{old} -> {tids}），作废这一根的旧产物重跑")
            _wipe_track_products(tdst)

        run_step([str(PY_GVHMR), str(SRC / "prep_boxes.py"),
                  "--video", str(proc), "--output-root", str(troot),
                  "--merge-tids", ",".join(str(t) for t in tids),
                  "--tracks-from", str(allp)],
                 f"{far} 检测框", 2, on_prog)
        if "rtmpose" in want:
            run_step([str(PY_MAIN), str(SRC / "run_rtmpose.py"),
                      "--src", str(tdst), "--dst", str(troot), "--video", str(proc)],
                     f"{far} RTMPose", 3, on_prog)
        if "gvhmr" in want:
            run_step([str(PY_GVHMR), str(SRC / "run_gvhmr.py"),
                      "--video", str(proc), "-s", "--output_root", str(troot)],
                     f"{far} 推理", 4, on_prog)
        if "export" in want:
            run_step([str(PY_GVHMR), str(SRC / "export_gvhmr.py"),
                      str(tdst / "hmr4d_results.pt"), "--spaces", str(exp["spaces"])],
                     f"{far} 导出 npz", 5, on_prog)
        # 凭证最后写，而且自己保证目录在：写在这之前的话，一次失败的运行会留下
        # "已经是这个分组"的假象，下次直接跳过——而盘上其实什么都没有。
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(tids), encoding="utf-8")
    ctx["merged_groups"] = [g["gid"] for g in groups]


def step_merge(raw: Path, name: str, ctx: dict, on_prog) -> None:
    """「归并人物」这一步：同一个人被切成的几段合起来跑一遍（见 _run_merged_groups）。

    和「多人轨迹」一样是**干净地跑完**：某一组失败不该把整条素材判死，也不该挡住
    后面几组。
    """
    try:
        _run_merged_groups(ctx, on_prog)
    except Exception as exc:                           # noqa: BLE001
        log(f"    ⚠ 归并失败（主轨迹的产物不受影响）：{exc}")


def step_tracks(raw: Path, name: str, ctx: dict, on_prog) -> None:
    """「多人轨迹」这一步：其余轨迹各跑一遍（见 `_run_extra_tracks`）。

    **必须是干净地跑完**：某一条轨迹失败不该把整条素材判死（主轨迹的产物早就
    在盘上了），也不该挡住后面几条。
    """
    try:
        _run_extra_tracks(ctx, on_prog)
    except Exception as exc:                           # noqa: BLE001
        log(f"    ⚠ 其余轨迹失败（主轨迹的产物不受影响）：{exc}")


def step_rtmpose(raw: Path, name: str, ctx: dict, on_prog) -> None:
    run_step([str(PY_MAIN), str(SRC / "run_rtmpose.py"),
              "--src", str(ctx["rt_dir"]), "--dst", str(ctx["rt_root"]),
              "--video", str(ctx["proc_video"])]
             + (["--overwrite"] if ctx["force"] else []),
             "RTMPose 提取", 3, on_prog)


def step_gvhmr(raw: Path, name: str, ctx: dict, on_prog) -> None:
    run_step([str(PY_GVHMR), str(SRC / "run_gvhmr.py"),
              "--video", str(ctx["proc_video"]), "-s",
              "--output_root", str(ctx["rt_root"])],
             "GVHMR/RTMPose", 4, on_prog)


def step_export(raw: Path, name: str, ctx: dict, on_prog) -> None:
    exp = normalize_step_opts("export", (ctx.get("step_opts") or {}).get("export"))
    run_step([str(PY_GVHMR), str(SRC / "export_gvhmr.py"),
              str(ctx["rt_dir"] / "hmr4d_results.pt"),
              "--spaces", str(exp["spaces"])],
             "导出 npz", 5, on_prog)


def _ensure_mesh(ctx: dict, on_prog) -> None:
    """渲染之前确保网格文件在。不在就用同一份 .pt 补一份（20~60 秒）。

    网格（vertices）占了结果 npz 的 99.5%，而**只有渲染读它**，所以它平时不
    落盘（见 mesh_npz.py 和 export_gvhmr.py）。既然平时不在，"渲染时补回来"
    就必须是渲染路径的一部分——否则第一次渲染会直接失败，而用户没做错任何事。

    补法就是重跑一次导出：它会重建网格、顺手把那份小的 npz 也刷新一遍
    （内容相同）。只补缺的那几个坐标系。
    """
    rt = ctx["rt_dir"]
    need = [s for s in ("incam", "global")
            if (rt / f"hmr4d_results_{s}.npz").exists()
            and not mesh_path_for(rt / f"hmr4d_results_{s}.npz").exists()]
    if not need:
        return
    if not (rt / "hmr4d_results.pt").exists():
        raise RuntimeError(f"要补网格，但 {rt / 'hmr4d_results.pt'} 不在——"
                           f"网格由它重建，这份丢了就只能重跑 GVHMR 推理")
    log(f"    网格不在盘上（{'、'.join(need)}）——它只有渲染要用，平时不落盘。"
        f"先从 hmr4d_results.pt 补一份（约 20~60 秒，一条素材只付这一次）")
    run_step([str(PY_GVHMR), str(SRC / "export_gvhmr.py"),
              str(rt / "hmr4d_results.pt"), "--spaces", ",".join(need)],
             "补网格", 6, on_prog)


def step_render(raw: Path, name: str, ctx: dict, on_prog) -> None:
    _ensure_mesh(ctx, on_prog)
    rt_i = ctx["rt_dir"] / "hmr4d_results_incam.npz"
    rt_g = ctx["rt_dir"] / "hmr4d_results_global.npz"
    quad = ctx["out_root"] / "quad.mp4"
    cmd = [str(PY_MAIN), str(SRC / "render_overlay.py"), str(rt_i),
           "--video", str(ctx["proc_video"]),
           # 音轨从**原片**取：渲染是裸帧管道不带音频，而预处理产物本身也没音轨
           "--audio", str(raw), "--layout", "quad", "-o", str(quad)]
    if rt_g.exists():
        cmd += ["--global-npz", str(rt_g)]
    # 没有世界系那份就不传 —— render_overlay 自己会把 quad 降级成 sidebyside
    # 并打印原因（见它那句"需要 --global-npz，已降级"）。**不能不判断就传**：
    # 传一个不存在的路径会让 np.load 直接抛异常，而真正的信息是"这份没导"。
    if not rt_g.exists():
        log("    （没有世界系 npz：quad 会降级成 sidebyside。需要的话把「导出 npz」"
            "的坐标系设成「两个都要」重跑那一步）")
    run_step(cmd, "渲染 quad", 6, on_prog)
    ctx["rendered"].append(quad)


STEP_FN = {
    "preprocess": step_preprocess,
    "boxes": step_boxes,
    "rtmpose": step_rtmpose,
    "gvhmr": step_gvhmr,
    "export": step_export,
    "merge": step_merge,
    "tracks": step_tracks,
    "render": step_render,
}


# ---------------------------------------------------------------------------
# 单条视频
# ---------------------------------------------------------------------------
def resolve_ctx(name: str, steps: list[str] | None = None) -> dict:
    """按素材短名把路径都找出来：预处理产物、rt_dir、世界系/相机系 npz。

    **路径约定只有这一处**——管线步骤和渲染预览都从这里拿，不各写一份
    （两份的后果是"预览渲染的是另一份 npz"这种查半天的怪事）。
    """
    out_root = OUTPUTS / name
    rt_root = out_root / "gvhmr_rtmpose"
    # 每次都从磁盘 glob，而不是依赖第 1 步刚跑过——用户可以只选后面几步，
    # 产物已经在盘上了。
    cands = sorted(PROCESSED.glob(f"{name}_*p*.mp4"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise RuntimeError(
            f"找不到预处理产物 {PROCESSED}/{name}_*p*.mp4；"
            "把 preprocess 也选上，或先单独跑一次")
    proc = cands[-1]
    # steps 只用来判断"其余轨迹要不要跟着跑"（见 _run_extra_tracks）——
    # 它不影响主线：主线该跑哪几步由调用方自己的循环决定。
    return {"out_root": out_root, "rt_root": rt_root, "proc_video": proc,
            "rt_dir": rt_root / proc.stem, "steps": list(steps or [])}


def process_one(raw: Path, steps: list[str], force: bool = False,
                step_opts: dict | None = None) -> dict:
    name = ascii_slug(raw.stem)
    out_root = OUTPUTS / name
    out_root.mkdir(parents=True, exist_ok=True)

    result = {"name": name, "orig": raw.stem, "ok": False, "videos": [],
              "error": None, "steps_done": []}

    total = len(steps)
    t0 = time.time()
    # `step_opts`（各步的可调参数）跟着 ctx 一路传下去：resolve_ctx 每次只
    # update 它自己那几个键，所以这里放进去的会一直留着。
    ctx: dict = {"force": force, "out_root": out_root,
                 "rt_root": out_root / "gvhmr_rtmpose", "rendered": [],
                 "step_opts": normalize_all_step_opts(step_opts)}

    for i, sid in enumerate(steps, 1):
        emit(kind="step", index=i, total=total, id=sid, name=STEP_NAME[sid],
             state="start")
        log(f"  [{i}/{total}] {STEP_NAME[sid]}")

        if sid != "preprocess":
            ctx.update(resolve_ctx(name, steps))

        st = time.time()
        step_fn = STEP_FN[sid]
        step_fn(raw, name, ctx, lambda f, _i=i: emit(kind="progress", index=_i, frac=f))
        el = time.time() - st
        result["steps_done"].append(sid)
        emit(kind="step", index=i, total=total, id=sid, name=STEP_NAME[sid],
            state="done", elapsed=round(el, 2))
        log(f"       完成 {el:.1f}s")

    # 其余轨迹**不再是自动的**，它是勾出来的「多人轨迹」那一步（见 step_tracks）。
    # 原来挂在这里自动跑，代价是乘性的——一条切镜头碎的素材能跟出二十几条轨迹，
    # 于是"跑一条视频"变成六七个小时、几十 GB，而每条只覆盖百分之几。
    result.update(ok=True, elapsed=time.time() - t0, videos=ctx["rendered"])
    return result


# ---------------------------------------------------------------------------
# 渲染预览（不写正式 quad.mp4）
# ---------------------------------------------------------------------------
def render_preview(args) -> int:
    """按给定的格子/时间段渲一段对照视频。

    **和管线里的 render 步骤共用 resolve_ctx 和 render_overlay.py**，区别只在
    产物落哪：预览是临时产物，参数可以随便调、随时删，不该覆盖正式 quad.mp4。
    参数从命令行来（作业队列拼好传进来），所以这里不做"默认四格"的假设——
    默认值在 --views/--layout 那边（render_overlay 自己知道）。
    """
    videos = sorted(VIDEOS.glob("*.mp4"))
    if args.only:
        videos = [v for v in videos if args.only in v.name]
    if len(videos) != 1:
        log(f"--only {args.only!r} 匹配到 {len(videos)} 个视频，渲染预览要精确到一条")
        emit(kind="failed", message=f"匹配到 {len(videos)} 个视频，无法确定渲哪一条")
        return 1
    raw = videos[0]
    name = ascii_slug(raw.stem)

    try:
        ctx = resolve_ctx(name)
    except RuntimeError as exc:
        log(f"  ✗ {exc}")
        emit(kind="failed", message=str(exc))
        return 1

    rt = ctx["rt_dir"]
    incam = rt / "hmr4d_results_incam.npz"
    glob_npz = rt / "hmr4d_results_global.npz"
    if not incam.exists():
        msg = f"没有重建结果 {incam}，先跑 GVHMR 推理 + 导出 npz"
        log(f"  ✗ {msg}")
        emit(kind="failed", message=msg)
        return 1

    # 网格平时不落盘（只有渲染读它），所以预览这条路上也要补——这是界面里
    # "渲一条预览"的入口，用户不该因为"网格没存"而看到失败。
    _ensure_mesh(ctx, lambda f: emit(kind="progress", index=1, frac=f))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY_MAIN), str(SRC / "render_overlay.py"), str(incam),
           "--video", str(ctx["proc_video"]),
           "--audio", str(raw),          # 音轨从原片取（预处理产物没有音轨）
           "-o", str(out),
           "--fps", "30"]
    if glob_npz.exists():
        cmd += ["--global-npz", str(glob_npz)]
    for flag, val in (("--views", args.views), ("--grid", args.grid),
                      ("--cell-h", args.cell_h), ("--frames", args.frames)):
        if val:
            cmd += [flag, str(val)]

    emit(kind="step", index=1, total=1, id="render", name="渲染预览", state="start")
    t0 = time.time()
    try:
        run_step(cmd, "渲染预览", 1, lambda f: emit(kind="progress", index=1, frac=f))
    except RuntimeError as exc:
        log(f"  ✗ {exc}")
        emit(kind="failed", message=str(exc)[:500])
        return 1
    emit(kind="step", index=1, total=1, id="render", name="渲染预览", state="done",
         elapsed=round(time.time() - t0, 2))
    log(f"预览写出：{out}")
    return 0


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="批量跑完整管线")
    ap.add_argument("--only", default=None, help="只处理文件名含该子串的视频")
    ap.add_argument("--steps", default=None,
                    help=f"逗号分隔的步骤 id，可选：{','.join(STEP_ORDER)}。"
                         f"默认 {','.join(DEFAULT_STEPS)}"
                         f"（不含 tracks、render 这两步要手动勾）")
    ap.add_argument("--skip-render", action="store_true",
                    help="等价于默认行为（render 本来就不跑），保留是为了兼容旧命令")
    ap.add_argument("--force", action="store_true", help="忽略已有产物重算")
    # 「多人轨迹」这一步的两个闸（见 TRACK_OPTS / _run_extra_tracks）。命令行
    # 给默认值，作业队列按界面上的输入覆盖。**0 表示"用默认值"**：argparse 的
    # 数值默认值没法区分"没给"和"给了 0"，而这里的 0 恰好不是合法取值。
    ap.add_argument("--track-min-coverage", type=float, default=None,
                    help="「多人轨迹」：只跑占比不低于这个数的轨迹（默认 0.20）")
    ap.add_argument("--track-max-tracks", type=int, default=None,
                    help="「多人轨迹」：最多跑几条（默认 3）")
    ap.add_argument("--track-only", default=None, metavar="TIDS",
                    help="「多人轨迹」：只跑这几条轨迹（逗号分隔）。"
                         "「人物」面板每一行的「重建」按钮走的就是它")
    ap.add_argument("--export-spaces", default=None,
                    help="「导出 npz」：导出哪些坐标系——incam（默认）/ global / "
                         "incam,global")
    # 渲染预览：作业队列用这几个参数拼出一次临时渲染（见 render_preview）。
    # 它们是**透传**给 render_overlay.py 的，这里不解释语义——两处各解释一遍
    # 迟早会不一致。
    ap.add_argument("--render-preview", action="store_true",
                    help="渲染一段对照预览（不写正式 quad.mp4）")
    ap.add_argument("--out", default=None, help="--render-preview 的产物路径")
    ap.add_argument("--views", default=None, help="透传给 render_overlay.py")
    ap.add_argument("--grid", default=None, help="透传给 render_overlay.py")
    ap.add_argument("--cell-h", default=None, help="透传给 render_overlay.py")
    ap.add_argument("--frames", default=None, help="透传给 render_overlay.py")
    args = ap.parse_args()

    if args.render_preview:
        if not args.out:
            log("--render-preview 需要 --out")
            return 2
        return render_preview(args)

    if args.steps:
        requested = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = [s for s in requested if s not in STEP_FN]
        if unknown:
            log(f"未知步骤：{', '.join(unknown)}\n可选：{', '.join(STEP_ORDER)}")
            return 2
        if not requested:
            log("--steps 是空的，没有要跑的步骤")
            return 2
        steps = [s for s in STEP_ORDER if s in requested]     # 强制规范顺序
    else:
        steps = list(DEFAULT_STEPS)

    videos = sorted(VIDEOS.glob("*.mp4"))
    if args.only:
        videos = [v for v in videos if args.only in v.name]
    if not videos:
        log(f"没找到视频：{VIDEOS}")
        return 1

    log(f"待处理 {len(videos)} 个视频，步骤：{' -> '.join(steps)}")
    log(f"  GVHMR 用 {PY_GVHMR}")
    log(f"  其余用 {PY_MAIN}")
    log("")

    done, failed = [], []
    for i, v in enumerate(videos, 1):
        log(f"[{i}/{len(videos)}] {v.stem}")
        log("-" * 66)
        try:
            r = process_one(v, steps, args.force, {
                "tracks": {"min_coverage": args.track_min_coverage,
                           "max_tracks": args.track_max_tracks,
                           "only": args.track_only or ""},
                "export": {"spaces": args.export_spaces},
            })
            done.append(r)
            log(f"  完成，用时 {r['elapsed']/60:.1f} 分钟\n")
        except Exception as exc:
            log(f"  ✗ 失败：{exc}\n")
            failed.append((v.stem, str(exc)))
            emit(kind="failed", orig=v.stem, message=str(exc)[:500])

    log("=" * 66)
    log(f"成功 {len(done)} / 失败 {len(failed)}")
    for r in done:
        log(f"  ✓ {r['name']}  ({r['elapsed']/60:.1f} 分钟)")
        for vid in r["videos"]:
            log(f"      {vid}")
    for nm, err in failed:
        log(f"  ✗ {nm}: {err[:200]}")
    log(f"\n产物根目录：{OUTPUTS}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
