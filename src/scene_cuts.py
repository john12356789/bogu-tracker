"""切镜头检测——给时间轴画虚线，也给 to_funscript 提供跨切点处理的边界。

为什么要单独一份切点清单
------------------------
跨过一次硬切，**世界系的绝对位置就不可观测了**：GVHMR 的世界系原点与朝向由
相机轨迹推出来，镜头一换就重新定位，于是同一个静止的人也会"瞬移"。实测这个
台阶进了 funscript 就是一次满行程撞击——一阶高通对阶跃 S 的响应，第一个样本
就输出 `a·S`（fc=0.5Hz、30fps 时 a=0.9006），之后以 0.32s 的时间常数衰减。
更阴的是它还会污染 `--fit` 的分位统计，把全片比例尺抬高、所有真实动作压小。

所以第一件事是知道切在哪一帧。**检测跑在预处理产物上**，因为整条管线（npz 的
帧号、2D 关键点、渲染）都是按预处理视频的帧空间索引的；跑原片会在源帧率不是
30 的素材上对不上号。

判据
----
逐帧降采样到 96×96 灰度（ffmpeg 管道流式，不把帧读进内存），算四个量：

  d_hist  相邻帧灰度直方图（64 箱）L1 距离的一半 ∈ [0,1] —— "画面组成变了多少"
  d_sust  前后各 W 秒窗内**平均直方图**的 L1 距离一半 ∈ [0,1] —— "变化留住了没有"
  d_prom  d_hist 相对邻域中位数的倍数 —— "这个变化有多孤立"
  d_gray  相邻帧逐像素灰度差的**均值** / 255   ∈ [0,1]（只进 score 与报告）
  d_frac  相邻帧灰度变化**超过 32 级的像素占比**  ∈ [0,1]（只进报告）

  score = d_hist + d_gray ∈ [0,2]，另外要求 d_sust ≥ SUST_MIN 且 d_prom ≥ PROM_MIN

**判"切没切"的是 d_prom，不是逐像素差。** 这条是踩过两次才定下来的：

- 一开始用 `mean|Δ像素| ≥ 0.12` 当闸门。它把**远近镜头切换全部漏掉**——实测
  BV1VVhk6pEMQ 里那 23 个（都肉眼确认过：中景切全景、同一个背板同一套打光）
  的 `mean|Δ|` 只有 **0.056~0.062**。原因是均值被大面积不变的背景稀释了：
  只有主体那块被放大/替换，背景还是那面墙。**这个漏洞就是"有些镜头切换认不出来"。**
- 改成"变化面积"（灰度差 >32 级的像素占比 ≥ 0.12）之后，那些真切换进来了
  （面积 0.127~0.146），但**淡入、溶解、模糊转场进得更多**（面积 0.33~0.34），
  边缘只有 0.02，分不开。连"亮度整体变化"这一类都要靠别的量兜。

  d_prom 一次把两类都解决：**硬切是单帧事件**——它那一帧的直方图变化远大于
  邻域；淡入/溶解/模糊是**连续几十帧都在变**，邻域的中位数本身就高，倍数上不去。
  实测（下面注释里有具体数）真切换 8~25 倍，渐变类 1.5~4 倍。

**d_sust 挡"瞬变"**（爆闪灯、单帧噪声）：那类 d_prom 很高（孤立），但变化留不住。
两个闸门互补，缺一个就在某一类上出错。

d_gray / d_frac 现在只进 score 与报告，不做闸门——留着是为了报告里能复核
"这一帧到底长什么样"。

阈值是**全片自适应**的：`thr = max(abs_min, 中位数 + k·MAD)`。固定阈值在不同
素材上不是漏就是误报——一条本身剪得碎的快剪，和一个长镜头静态画面，分数分布
完全不同。中位数和 MAD 对"少数几个大尖峰"免疫（均值和标准差会被那几帧拉走）。

检测不到切点很正常（大量素材本来就是单镜头），返回空清单即可，不是错误。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SIZE = 96          # 检测用的边长。64 在暗部噪点上不稳，128 收益已尽（测得分数几乎相同）
# 输出侧的帧率模式，**必须显式给**，不能吃默认值。
#
# ffmpeg 默认的 `-fps_mode auto`：输出格式（rawvideo）不支持可变帧率时，它会把
# 时间轴**补成匀速**——源里空着的帧位被复制出来的帧填上。实测本素材的源在
# 第 1943 个包之后有 4 个空帧位（原始文件自带，预处理照搬了过来），默认参数因此
# 解出 **2450** 帧，而重建那条线（npz 的帧数、2D 关键点、渲染都按解码出的呈现
# 顺序索引）只有 **2444** 帧。差 6 帧就是 0.2 秒，而且缺口之后的切点帧号会整体偏。
#
# passthrough 是 ffmpeg 5+ 的写法，老版本要用 -vsync 0（同一个东西的旧名），
# 两个都试一遍；连不上就说明这个 ffmpeg 太老，得换。
FPS_MODES = (("-fps_mode", "passthrough"), ("-vsync", "0"))

BINS = 64          # 直方图箱数
VERSION = 1

# 阈值参数。abs_min 是"绝对下限"——全片分数都极低（静物长镜头）时，
# 中位数+6σ 会退化成"总有一个最大的"，把噪声报成切点。
ABS_MIN = 0.18
# 持续差异的闸门。定在 0.10 是量出来的：实测硬切在 0.24~0.51，
# 而运镜、快动作、闪光都在 0.11 以下。挡的是"只闪了一两帧"。
SUST_MIN = 0.10
# 突出度闸门：d_hist 相对**邻域中位数**的倍数。
# 这是挡渐变的真正那一道（见模块开头：逐像素差和变化面积都试过，都会漏远近切换
# 或者放进一堆淡入/溶解）。实测单帧硬切 8~25 倍，连续几十帧的渐变 1.5~4 倍。
PROM_MIN = 5.0
PROM_WIN = 0.5          # 秒。算邻域中位数的半窗
PROM_FLOOR = 0.01       # 邻域中位数的下限，防止静物素材上除出天文数字
CHANGE_LEVEL = 32       # 一个像素算"变了"的门槛（0~255），只用于报告里的 d_frac
K_MAD = 6.0
# 切点之间的最小间隔（秒）。**不能大**：真实素材里有连着的快切，
# 0.4s 会把它们吃掉——实测 BV1AtKS6fEAP（漫展直拍）有 3 个分数 0.66 的候选
# 被 0.4s 挤掉。硬切是**单帧**事件，0.2s（30fps 下 6 帧）足够分开一个转场，
# 又不会吃掉相邻镜头。
MIN_GAP = 0.2
SUST_WIN = 0.25    # 秒。前后各取这么长的窗比平均直方图
# scores_from 输出的列。用具名常量而不是下标字面量：加这一列的时候
# pick_cuts/CLI 里的 0/1/2/3 全都得改一遍，那种改动最容易漏。
C_HIST, C_SUST, C_PROM, C_GRAY, C_FRAC, C_SCORE = range(6)

# 可调的判据：**只此一份定义**。CLI 的默认值、Web 的 /api/cuts/params、以及落盘
# 的 params 都从这里来——三处各写一遍必然漂移，而漂移的表现是"面板上显示 5，
# 实际按 6 在跑"，极难查。
#
# 这六个数就是"灵敏不灵敏"的全部旋钮：阈值 = max(abs_min, 中位 + k·σ)，
# 三个闸门是 sust_min / prom_min / 面积（由 change_level 定义），
# min_gap 决定多近的两个切点会被合并成一个。
TUNABLE: dict[str, float | int] = {
    "abs_min": ABS_MIN, "k": K_MAD, "sust_min": SUST_MIN,
    "prom_min": PROM_MIN, "min_gap": MIN_GAP, "change_level": CHANGE_LEVEL,
}

# 取值域。夹取而不是报错——"调过头"是滑杆上随手就会发生的事，为了一个越界的
# 数字让整次重测失败（而重测要解码整条视频）是不划算的。
LIMITS: dict[str, tuple[float, float]] = {
    "abs_min": (0.0, 2.0), "k": (0.0, 20.0), "sust_min": (0.0, 1.0),
    "prom_min": (0.0, 50.0), "min_gap": (0.0, 2.0), "change_level": (1.0, 255.0),
}


def defaults() -> dict:
    """判据的默认值（面板要显示的）。"""
    return dict(TUNABLE)


def normalize(params: dict | None) -> dict:
    """外部传进来的判据 -> 合法的一套。缺的补默认值，多的丢掉，越界的夹回来。"""
    out = dict(TUNABLE)
    for key, raw in (params or {}).items():
        if key not in TUNABLE or raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        lo, hi = LIMITS[key]
        val = min(max(val, lo), hi)
        out[key] = int(round(val)) if isinstance(TUNABLE[key], int) else val
    return out


def iter_gray(video: str | Path, size: int = SIZE,
              mode: tuple[str, str] = FPS_MODES[0]):
    """流式产出降采样灰度帧 (size,size) uint8。

    走 ffmpeg 管道而不是 cv2.VideoCapture：解码 + 缩放全在 ffmpeg 里多线程做完，
    一条 5 分钟的 1280p 素材几秒就能过完，Python 只拿到 96×96 的字节流。
    （run_rtmpose 那次 OOM 就是因为在 Python 侧攒帧——这里一个字节都不留。）

    `mode` 见 FPS_MODES：**不显式指定就会多出帧来**。

    stderr 接到临时文件而不是 PIPE：PIPE 在 ffmpeg 话多时会写满管道缓冲区，
    两边一起卡死。临时文件没有这个上限，出错时还能把尾部内容读出来。
    """
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
        "-vf", f"scale={size}:{size}:flags=area",
        *mode,
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=1 << 20)
    n = size * size
    try:
        while True:
            buf = proc.stdout.read(n)
            if not buf or len(buf) < n:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(size, size)
    finally:
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            err.seek(0)
            tail = err.read().decode("utf-8", "replace").strip()[-500:]
            raise RuntimeError(f"ffmpeg 解码失败（退出码 {rc}）：\n{tail}")
        err.close()


def features(video: str | Path, size: int = SIZE, quiet: bool = False,
             change_level: int = CHANGE_LEVEL
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一趟解码，返回 (直方图 (T,BINS) float32, 灰度差均值 (T,), 变化面积 (T,))。

    直方图整条留着（9000 帧 × 64 箱 × 4 字节 = 2.3 MB），因为 d_sust 要拿窗口
    平均，流式算不了。像素帧本身仍然一个都不留。

    帧率模式按 FPS_MODES 逐个试：老 ffmpeg 不认 -fps_mode，会立刻非零退出
    （一帧都不会产出），所以退回去试旧写法是安全的。
    """
    last: Exception | None = None
    for mode in FPS_MODES:
        try:
            return _features(video, size, mode, quiet, change_level)
        except RuntimeError as exc:            # noqa: PERF203
            last = exc
            print(f"⚠ 帧率模式 {' '.join(mode)} 不可用，换下一个：{exc}",
                  file=sys.stderr)
    raise last if last else RuntimeError("ffmpeg 解不出帧")


def _features(video: str | Path, size: int, mode: tuple[str, str], quiet: bool,
              change_level: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hists: list[np.ndarray] = []
    dgray: list[float] = []
    dfrac: list[float] = []
    prev_gray: np.ndarray | None = None
    for i, f in enumerate(iter_gray(video, size, mode)):
        h = np.bincount(f.ravel(), minlength=256).astype(np.float32)
        h = h.reshape(BINS, 256 // BINS).sum(axis=1)
        h /= max(float(h.sum()), 1.0)
        hists.append(h)
        if prev_gray is None:
            dgray.append(0.0)
            dfrac.append(0.0)
        else:
            diff = np.abs(f.astype(np.int16) - prev_gray)
            dgray.append(float(diff.mean()) / 255.0)
            dfrac.append(float((diff > change_level).mean()))
        prev_gray = f.astype(np.int16)
        if not quiet and i and i % 3000 == 0:
            print(f"  切点检测 {i} 帧…", file=sys.stderr)
    return (np.asarray(hists, dtype=np.float32),
            np.asarray(dgray, dtype=np.float64),
            np.asarray(dfrac, dtype=np.float64))


def scores_from(hists: np.ndarray, dgray: np.ndarray, dfrac: np.ndarray, fps: float,
                win_s: float = SUST_WIN) -> np.ndarray:
    """特征 -> (T,6)：[d_hist, d_sust, d_prom, d_gray, d_frac, score]。第 0 帧全 0。

    d_sust 用前缀和求窗均值，O(T·BINS)——逐帧去切片求平均也行，但那是 O(T·W·BINS)，
    在 9000 帧的素材上差一个量级，而这段代码以后会被 Web 端点在一次 HTTP 请求里调用。

    d_prom 用滑窗**中位数**做邻域基准，不用均值：窗里正好装着另一个切点是常事
    （连着的快切），均值会被它抬起来，中位数不会（两个点改变不了 31 个数的中位）。
    """
    t_n = len(hists)
    out = np.zeros((t_n, 6), dtype=np.float64)
    if t_n < 2:
        return out
    w = max(1, int(round(win_s * fps)))
    cs = np.cumsum(hists.astype(np.float64), axis=0)          # (T,BINS)

    def win_mean(a: int, b: int) -> np.ndarray | None:
        a, b = max(0, a), min(t_n, b)
        if b <= a:
            return None
        s = cs[b - 1] - (cs[a - 1] if a > 0 else 0.0)
        return s / (b - a)

    for t in range(1, t_n):
        out[t, C_HIST] = 0.5 * float(np.abs(hists[t] - hists[t - 1]).sum())
        out[t, C_GRAY] = dgray[t]
        out[t, C_FRAC] = dfrac[t]
        pre, post = win_mean(t - w, t), win_mean(t, t + w)
        if pre is not None and post is not None:
            out[t, C_SUST] = 0.5 * float(np.abs(pre - post).sum())

    pw = max(1, int(round(PROM_WIN * fps)))
    d = out[:, C_HIST]
    pad = np.pad(d, pw, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * pw + 1)   # (T, 2pw+1)
    out[:, C_PROM] = d / np.maximum(np.median(win, axis=1), PROM_FLOOR)

    out[:, C_SCORE] = out[:, C_HIST] + out[:, C_GRAY]
    return out


def pick_cuts(scores: np.ndarray, fps: float, abs_min: float = ABS_MIN,
              k: float = K_MAD, min_gap: float = MIN_GAP,
              sust_min: float = SUST_MIN, prom_min: float = PROM_MIN
              ) -> tuple[list[int], dict]:
    """分数 -> 切点帧号。返回 (帧号, 阈值信息)。

    三道门，各挡一类，缺一个就在某类素材上误报或漏报（都是实测出来的，见模块开头）：
      瞬时分数 ≥ 自适应阈值   挡幅度不够的
      持续差异 ≥ sust_min     挡只闪一两帧的（闪光、爆闪灯）
      突出度   ≥ prom_min     挡渐变（淡入淡出、溶解、模糊转场）—— 单帧事件才过

    贪心：按分数从高到低取，与已选中的间隔小于 min_gap 的丢掉。这样一次转场
    里只有最强的那个存活，而且**最强的那帧**才有意义——它才是新镜头的第一帧。
    min_gap 而不是"把连续的合并成一段再取中点"：转场是瞬间的，取中点等于把
    边界挪到镜头里面去，跨切处理的窗口就偏了。
    """
    y = scores[:, C_SCORE]
    if len(y) < 3:
        return [], {"thr": None, "median": 0.0, "sigma": 0.0}
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    sigma = 1.4826 * mad                      # MAD -> 正态等效标准差
    thr = max(abs_min, med + k * sigma)

    cand = np.flatnonzero((y >= thr) & (scores[:, C_SUST] >= sust_min)
                          & (scores[:, C_PROM] >= prom_min))
    gap = max(1, int(round(min_gap * fps)))
    kept: list[int] = []
    for i in cand[np.argsort(-y[cand])]:
        if all(abs(int(i) - j) >= gap for j in kept):
            kept.append(int(i))
    kept.sort()
    return kept, {"thr": float(thr), "median": med, "sigma": sigma}


def detect_full(video: str | Path, size: int = SIZE, params: dict | None = None,
                quiet: bool = False) -> tuple[dict, np.ndarray]:
    """完整一趟：解码 -> 打分 -> 出切点。返回 (可落盘的字典, 分数矩阵)。

    `params` 是判据（见 TUNABLE），缺的用默认值补、越界的夹回来。

    **帧率从 ffprobe 取，不猜**：切点帧号要送进 to_funscript，而它按 `at =
    round(i*1000/fps)` 写时间轴，fps 错了时间就整体偏掉（29.97 与 30 差 0.1%，
    在 81s 素材上是 82ms）。
    """
    import preprocess                              # 只为复用它那份 ffprobe 封装

    pr = normalize(params)
    info = preprocess.probe(str(video))
    hists, dgray, dfrac = features(video, size, quiet=quiet,
                                   change_level=int(pr["change_level"]))
    scores = scores_from(hists, dgray, dfrac, info["fps"])
    n = len(scores)
    cuts, thr_info = pick_cuts(scores, info["fps"], pr["abs_min"], pr["k"],
                               pr["min_gap"], pr["sust_min"], pr["prom_min"])

    if info["frames"] and abs(n - info["frames"]) > 2:
        # 走到这里说明帧率模式没起作用（见 FPS_MODES）——切点帧号会和重建那边
        # 的帧空间错位，**不是可以忽略的告警**：缺口之后的切点会整体偏。
        print(f"⚠ 解出 {n} 帧，容器里 {info['frames']} 帧。多出来的是 ffmpeg 补的，"
              "切点帧号会与重建（npz/2D 关键点）对不上——别直接用这份清单。",
              file=sys.stderr)
    data = {
        "version": VERSION,
        "video": Path(video).name,
        "fps": float(info["fps"]),
        "n_frames": n,
        "cuts": cuts,
        # 每个切点的分数，便于核对（不是全部帧的分数，那个太大也没用）
        "cut_scores": [round(float(scores[c, C_SCORE]), 4) for c in cuts],
        # 每个切点的三项证据都留着："这个到底是不是切点"要能复核，
        # 只给一个总分等于让人只能选择相信
        "cut_evidence": [
            {"frame": int(c),
             "hist": round(float(scores[c, C_HIST]), 4),
             "sust": round(float(scores[c, C_SUST]), 4),
             "prom": round(float(scores[c, C_PROM]), 2),
             "gray": round(float(scores[c, C_GRAY]), 4),
             "frac": round(float(scores[c, C_FRAC]), 4)} for c in cuts],
        # 阈值信息留着：分数分布因素材而异，出问题时先看这几个数
        "threshold": round(thr_info["thr"], 4) if thr_info["thr"] is not None else None,
        "median": round(thr_info["median"], 4),
        "sigma": round(thr_info["sigma"], 4),
        # 落盘带上**这一次实际用的**判据：改了判据重测过之后，
        # 这份清单就该能自己说明它是按什么标准选出来的
        "params": {"size": size, "prom_win": PROM_WIN, **pr},
    }
    return data, scores


def detect(video: str | Path, params: dict | None = None, size: int = SIZE,
           quiet: bool = False) -> dict:
    """只要结果字典的入口（Web 端点用）。"""
    return detect_full(video, size, params, quiet)[0]


def save(data: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再改名：调用方可能在另一个进程里读它（Web 端点一次请求触发检测，
    # 前端同时在轮询），读到半截 JSON 会当成"没有切点"，然后又被下一次覆盖。
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def load(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cut_frames(path: str | Path, n_frames: int | None = None) -> list[int]:
    """读切点帧号，可选按 n_frames 校验/夹取。读不到就是空清单。

    长度对不上时**只夹取、不缩放**：帧数不一致说明这份清单是另一条预处理产物
    上测的（同一条素材重跑过不同分辨率），缩放的假设（帧号线性对应）在两份
    产物帧率不同时并不成立，猜一个不如老实丢掉越界的。
    """
    data = load(path)
    if not data:
        return []
    cuts = [int(c) for c in data.get("cuts", [])]
    if n_frames is not None:
        cuts = [c for c in cuts if 0 < c < n_frames]
    return cuts


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="切镜头检测（给时间轴与 to_funscript 用）")
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="写出 cuts.json；不给则只打印")
    ap.add_argument("--size", type=int, default=SIZE)
    # 判据的旗标直接从 TUNABLE 生成：加了新判据只要往那张表里添一行，
    # CLI / Web / 落盘三处一起就有——手写六遍迟早漏一个。
    for _name, _val in TUNABLE.items():
        _lo, _hi = LIMITS[_name]
        ap.add_argument("--" + _name.replace("_", "-"),
                        type=type(_val), default=_val,
                        metavar=f"[{_lo:g}~{_hi:g}]")
    ap.add_argument("--top", type=int, default=0,
                    help="另列出分数最高的 N 个候选（含被闸门挡掉的），用来核对漏检")
    ap.add_argument("--jpeg", default=None, metavar="DIR",
                    help="把每个切点的前后两帧导成 jpg，人工核对用")
    args = ap.parse_args(argv)

    if not args.out and not args.top and not args.jpeg:
        print("没给 --out/--top/--jpeg，什么都不会产出", file=sys.stderr)

    data, scores = detect_full(args.video, args.size,
                               {k2: getattr(args, k2) for k2 in TUNABLE})
    cuts = data["cuts"]
    print(f"{data['video']}  {data['n_frames']} 帧 @ {data['fps']:g}fps")
    print(f"分数分布：中位 {data['median']}  σ(MAD) {data['sigma']}  "
          f"阈值 {data['threshold']}")
    _pr = data["params"]
    print(f"判据：阈值 ≥ max({_pr['abs_min']:g}, 中位 + {_pr['k']:g}σ)  "
          f"持续 ≥ {_pr['sust_min']:g}  突出 ≥ {_pr['prom_min']:g}x  "
          f"最小间隔 {_pr['min_gap']:g}s  变化级别 {_pr['change_level']}")
    print(f"切点 {len(cuts)} 个：")
    for c, s in zip(cuts, data["cut_scores"]):
        print(f"  帧 {c:>6}  {c / data['fps']:8.3f}s  分数 {s}")

    if args.top:
        order = np.argsort(-scores[:, C_SCORE])[: args.top]
        print(f"\n分数最高的 {args.top} 帧：")
        for i in sorted(int(x) for x in order):
            h, su, pr, g, fr, s = scores[i]
            if i in cuts:
                mark = "  <= 已选"
            elif s >= (data["threshold"] or 1e9):
                why = "持续差异不够" if su < args.sust_min else "不够孤立（像是渐变）"
                mark = f"  <= 过阈值但{why}，挡掉"
            else:
                mark = ""
            print(f"  帧 {i:>6}  {i / data['fps']:8.3f}s  hist {h:.3f}  "
                  f"持续 {su:.3f}  突出 {pr:5.1f}x  gray {g:.3f}  面积 {fr:.3f}  "
                  f"合计 {s:.3f}{mark}")

    if args.out:
        print(f"\n写出 {save(data, args.out)}")

    if args.jpeg:
        d = Path(args.jpeg)
        d.mkdir(parents=True, exist_ok=True)
        for c in cuts:
            for k_, name in ((-1, "before"), (0, "after")):
                idx = c + k_
                if idx < 0:
                    continue
                out = d / f"{c:06d}_{name}.jpg"
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(args.video),
                     "-vf", f"select=eq(n\\,{idx})", "-vsync", "0", "-frames:v", "1",
                     "-q:v", "3", str(out)],
                    check=False)
        print(f"切点前后帧已导出到 {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
