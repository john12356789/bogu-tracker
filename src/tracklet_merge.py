"""把同一个人的多条碎轨迹归并成一条（tracklet stitching）。

为什么需要它
------------
ByteTrack 是**在线**跟踪：逐帧做 IoU 匹配，丢几帧（快动作、遮挡、切镜头）
轨迹就断了，重新检出时是一个新 tid。实测那条 5192 帧的双人素材里，两个人
各被切成 9~12 段。所以：

    一个人 ≠ 一条轨迹        一个人 = **一组**轨迹段

这里只解决"哪些段属于同一个人"。合并成一条框轨迹、重跑重建是下游的事
（`union_pick` + batch_run 的归并步骤）。

算法：一次前向链 + 一次补洞
---------------------------
1. **硬约束：同一组里任意两段的"同时在场"不许超过 OVERLAP_MAX 帧**（按 runs 判）。
   同一时刻只能在一个地方——这条不花钱、挡掉的错误配对最多，而且**违反它一定
   是 bug**，所以可以自动断言，不需要人工标注。留一点重叠是必要的：同一人身上
   开两条轨是 ByteTrack 的常见错误，那两条轨会有几帧重叠；而真两个人同框时，
   重叠是整段整段的，差着两个数量级。
2. **前向链**：按出现时间排序，每段去接"已经在它之前结束"的组里最像的那个。
   三个线索全是尺度无关量（除以框高），远近景都适用：
     · 位置：用前一段**末端**的框心速度外推到这一段的起点，看落点差几个框高；
     · 间隔：越长越可疑，但只是罚分不是闸（见 link_score）；
     · 体格：两边的框高相对差。**固定机位下同一个人的框高基本恒定**，所以它
       既是线索也是硬闸——差过 DH_MAX 直接不许并（这是"该分开"的那道闸）；
     · 速度连续性（权重最低，舞蹈本来就会急停反转）。
   一个人的段在时间上是一条链、没有分支，所以贪心链就是这个问题结构的正解，
   不需要 Hungarian/图划分。
3. **补洞**：前向链看不到"插进中间"的段，可它恰恰是 ByteTrack 最典型的错误形态
   ——丢号之后新号先出现、老号又续上（0~40 是 1、45~59 是 2、60~100 又是 1）。
   所以再走一遍：把还单着的段试着塞进某个组**两段 runs 之间的洞**里，条件是
   **两端**的几何都对得上。插进去的帧是真实检测，比原先那段插值补的框更可信。

边界（诚实说清楚）
------------------
没有外观特征可用（现有的 vit_features.pt 是 HMR2 的，为 mesh recovery 训的，
不是身份特征）。所以"体格接近的两个人交替出现"这种情形，几何上就是分不开——
输出只能是候选 + 置信度，**必须由人确认**。合并错了比不合并糟得多：动作会直接
变成另一个人的。

阈值都是先验值（见下面那组常量），实测再调。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# ---- 阈值。单位是"帧"或"框高"。都是先验值，但下面两个数是**实测定的**，
#      改之前先看 src/tests/test_tracklet_merge.py 里的用例 ----
GAP_MAX = 240         # 间隔的**硬上限**（8s）。超过它就别接了
GAP_SCALE = 90        # 间隔罚分的尺度：隔 90 帧罚满（见 W_GAP）
OVERLAP_MAX = 15      # 允许的重叠帧数（0.5s）。同一人身上开了两条轨时会有小重叠
DPOS_MAX = 1.0        # 外推落点差几个框高算"完全对不上"
DH_MAX = 0.35         # 框高相对差超过它就硬闸掉——不是同一个人
DV_MAX = 0.25         # 速度差（框高/帧）
W_POS, W_GAP, W_VEL = 0.55, 0.27, 0.18
SCORE_MIN = 0.5       # 合成得分低于它就不并
VEL_WINDOW = 15       # 外推用末端/首端各多少帧


# ---------------------------------------------------------------------------
# 段的规整
# ---------------------------------------------------------------------------
def _norm(seg: dict) -> dict | None:
    """bbx_all.json 的一段 -> 内部结构。字段缺了就丢（宁可少并，不许崩）。"""
    try:
        frames = [int(f) for f in (seg.get("frames") or [])]
        boxes = [[float(v) for v in b] for b in (seg.get("boxes") or [])]
    except (TypeError, ValueError):
        return None
    if not frames or len(frames) != len(boxes):
        return None
    order = sorted(range(len(frames)), key=lambda i: frames[i])
    frames = [frames[i] for i in order]
    boxes = [boxes[i] for i in order]
    runs: list[dict] = []
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[i - 1] + 1:
            runs.append({"i0": start, "i1": i - 1,
                         "f0": frames[start], "f1": frames[i - 1]})
            start = i
    return {"tid": int(seg["tid"]), "frames": frames, "boxes": boxes, "runs": runs}


def _span(s: dict) -> tuple[int, int]:
    return s["frames"][0], s["frames"][-1]


def _max_overlap(a: dict, b: dict) -> int:
    """两段同时被检出的最长帧数。**按 runs 判**，不是按首尾——一条 tid 可能中间
    断过，用首尾会把它两次出现之间的空档也算成"它在场"。"""
    m = 0
    for ra in a["runs"]:
        for rb in b["runs"]:
            m = max(m, min(ra["f1"], rb["f1"]) - max(ra["f0"], rb["f0"]) + 1)
    return max(0, m)


def _center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _height(b: list[float]) -> float:
    return max(1e-6, b[3] - b[1])


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _median(v: list[float]) -> float:
    if not v:
        return 0.0
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _slope(frames: list[int], values: list[float]) -> float:
    """最小二乘斜率（值/帧）。少于 2 个点，或者点全落在同一帧上，都是 0。"""
    n = len(frames)
    if n < 2:
        return 0.0
    mf = _mean([float(f) for f in frames])
    mv = _mean(values)
    den = sum((float(f) - mf) ** 2 for f in frames)
    if den <= 0:
        return 0.0
    return sum((float(frames[i]) - mf) * (values[i] - mv) for i in range(n)) / den


def _edge(seg: dict, run: dict, which: str) -> dict:
    """取某个 run 头/尾的样本（框心、框高、速度）。

    **窗口不卡在这个 run 里**，不够就往相邻的 run 借：一条 tid 自己也可能很碎
    （实测 tid 27 的第一段只有 3 帧），只看这 3 帧算出来的"体格"会把同一个人
    判成两个人。

    **框高用中位数**：那 3 帧是半身框，框高 552，而稳定值是 913——均值会被它们
    拉下来，中位数不会。实测就是这一条把同一个人的两段挡在了体格闸外面。

    **速度只在这个 run 内部拟合**：跨过断点拟合会把"两次出现之间的位移"当成
    速度，那是个假数。run 太短（< 4 帧）时给 0 = 不做外推，退化成"两端的框心
    直接比距离"，对碎轨迹来说这本来也是更诚实的判据。
    """
    lo, hi = run["i0"], run["i1"]
    if which == "head":
        lo, hi = run["i0"], min(len(seg["frames"]) - 1, run["i0"] + VEL_WINDOW - 1)
    else:
        lo, hi = max(0, run["i1"] - VEL_WINDOW + 1), run["i1"]
    bx = seg["boxes"][lo:hi + 1]
    hs = [_height(x) for x in bx]
    cs = [_center(x) for x in bx]
    si0, si1 = run["i0"], run["i1"]
    if si1 - si0 + 1 >= 4:
        rf = seg["frames"][si0:si1 + 1]
        rc = [_center(b) for b in seg["boxes"][si0:si1 + 1]]
        vel = (_slope(rf, [c[0] for c in rc]), _slope(rf, [c[1] for c in rc]))
    else:
        vel = (0.0, 0.0)
    return {"frame": seg["frames"][si1] if which == "tail" else seg["frames"][si0],
            "center": cs[-1] if which == "tail" else cs[0],
            "height": _median(hs),
            "vel": vel, "n": len(bx)}


def _clip(v: float, hi: float) -> float:
    return 0.0 if v <= 0 else (1.0 if v >= hi else v / hi)


def link_score(prev_tail: dict, next_head: dict,
               gap_scale: float = GAP_SCALE) -> tuple[float, dict] | None:
    """从前一段末尾外推到这个段开头的匹配分。返回 None = 撞上硬闸（体格差太多）。

    所有距离都除以框高，所以"人离镜头远近"不影响判据。

    **间隔是罚分不是闸**。原来是"超过 45 帧直接不接"，而实测那条有真值的素材
    （BV1RZY4zPEw2：甲 100% + 乙被切成 61%+33%）里，乙自己中间断了 **144 帧**
    ——用闸的话正确答案就在闸外面。间隔越长越可疑，但它该和位置、体格一起投票，
    而不是单独一票否决。
    """
    dh = abs(prev_tail["height"] - next_head["height"]) / max(prev_tail["height"],
                                                             next_head["height"])
    if dh > DH_MAX:
        return None
    dt = next_head["frame"] - prev_tail["frame"]
    px = prev_tail["center"][0] + prev_tail["vel"][0] * dt
    py = prev_tail["center"][1] + prev_tail["vel"][1] * dt
    href = max(prev_tail["height"], next_head["height"])
    dpos_rel = math.hypot(px - next_head["center"][0], py - next_head["center"][1]) / href
    dv = math.hypot(prev_tail["vel"][0] - next_head["vel"][0],
                    prev_tail["vel"][1] - next_head["vel"][1]) / href
    # dt < 0（重叠）不给间隔罚分：那是同一个人身上开了两条轨的形态，不是"隔了很久"
    pen = (W_POS * _clip(dpos_rel, DPOS_MAX)
           + W_GAP * _clip(dt / gap_scale, 1.0)
           + W_VEL * _clip(dv, DV_MAX))
    return max(0.0, 1.0 - pen), {"dpos": round(dpos_rel, 3), "dh": round(dh, 3),
                                 "dv": round(dv, 3), "gap": int(dt)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _conflict(a: dict, b: dict, overlap_max: int) -> bool:
    """两段"同时在场"得太久 → 一定是两个人。"""
    return _max_overlap(a, b) > overlap_max


def suggest(segments: list[dict], n_frames: int | None = None,
            gap_max: int = GAP_MAX, score_min: float = SCORE_MIN,
            overlap_max: int = OVERLAP_MAX) -> list[dict]:
    """bbx_all.json 的 segments -> 人物分组。

    返回 `[{"gid", "tids", "confidence", "runs", "n_frames", "coverage", "links"}]`。
    gid = 组内最小 tid（下游就用它当这条"人"的编号）。单段自成一组也返回——
    界面要完整列表才能显示"哪些还单着"。

    **允许小重叠**（≤ overlap_max 帧）：同一时刻只有一个人在场是对的，但同一
    个人身上开两条轨是 ByteTrack 的常见错误（实测 tid 23 还在跑的时候 tid 27
    已经在同一个人身上起来了）。所以判据是"重叠**不超过** overlap_max"，而不是
    一点都不许——真两个人同框时重叠是整段整段的，差着两个数量级。
    """
    items = [x for x in (_norm(s) for s in segments) if x is not None]
    items.sort(key=lambda x: (x["frames"][0], x["tid"]))

    groups: list[dict] = []
    for it in items:
        best = None
        for g in groups:
            if it["frames"][0] - g["last"] > gap_max:      # 负值 = 重叠，下面单独判
                continue
            if any(_conflict(it, s, overlap_max) for s in g["segs"]):
                continue
            tail = g["tail_run"]
            r = link_score(_edge(tail["seg"], tail, "tail"),
                           _edge(it, it["runs"][0], "head"))
            if r is None or r[0] < score_min:
                continue
            if best is None or r[0] > best[0]:
                best = (r[0], g, r[1])
        if best is None:
            groups.append({"segs": [it], "runs": list(it["runs"]),
                           "last": it["frames"][-1],
                           "tail_run": {"seg": it, **it["runs"][-1]},
                           "links": []})
        else:
            sc, g, detail = best
            g["segs"].append(it)
            g["runs"] = sorted(g["runs"] + list(it["runs"]), key=lambda r: r["f0"])
            g["last"] = max(g["last"], it["frames"][-1])
            g["tail_run"] = {"seg": it, **it["runs"][-1]}
            g["links"].append({"a": g["segs"][-2]["tid"], "b": it["tid"],
                               "score": round(sc, 3), **detail})

    _fill_holes(groups, gap_max, score_min, overlap_max)
    return [_finish(g, n_frames) for g in groups]


def _runs_of(g: dict) -> list[dict]:
    """组里全部 runs，每个都带上它属于哪一段。"""
    return [{"seg": s, **r} for s in g["segs"] for r in s["runs"]]


def _fill_holes(groups: list[dict], gap_max: int, score_min: float,
                overlap_max: int = OVERLAP_MAX) -> None:
    """把还单着的段塞进别的组两段 runs 之间的洞里（**两端**都要对得上）。

    这一段就是 ByteTrack 最典型的错误形态：丢号之后新号先出现、老号又续上。
    前向链只会把中间那个新号单独成组。插进去之后，那些帧用的是真实检测框，
    而不是原来那段拿前后帧插值补的框。

    只处理"两边都有"的洞；两端之外的（比组的第一段还早、比最后一段还晚）由
    前向链负责——按时间顺序处理时它们本来就会先/后接上。

    **洞可能在一段自己的内部**：一条 tid 中间断过（同一个号的两次 run），断口
    里插着的正是那个新号。所以候选宿主不能限定成"由多段组成的组"，只要它的
    runs 在那一段两边有就行。
    """
    singles = [g for g in groups if len(g["segs"]) == 1]
    merged = []
    for s in singles:
        seg = s["segs"][0]
        s0, s1 = _span(seg)
        best = None
        for g in groups:
            if g is s:
                continue
            if any(_conflict(seg, o, overlap_max) for o in g["segs"]):
                continue
            runs = _runs_of(g)
            before = [r for r in runs if r["f1"] < s0]
            after = [r for r in runs if r["f0"] > s1]
            if not before or not after:
                continue
            p_run = max(before, key=lambda r: r["f1"])
            n_run = min(after, key=lambda r: r["f0"])
            if s0 - p_run["f1"] > gap_max or n_run["f0"] - s1 > gap_max:
                continue
            r1 = link_score(_edge(p_run["seg"], p_run, "tail"),
                            _edge(seg, seg["runs"][0], "head"))
            r2 = link_score(_edge(seg, seg["runs"][-1], "tail"),
                            _edge(n_run["seg"], n_run, "head"))
            if r1 is None or r2 is None:
                continue
            sc = min(r1[0], r2[0])
            if sc < score_min:
                continue
            if best is None or sc > best[0]:
                best = (sc, g, r1[1], r2[1])
        if best is None:
            continue
        sc, g, d1, d2 = best
        g["segs"].append(seg)
        g["runs"] = sorted(g["runs"] + list(seg["runs"]), key=lambda r: r["f0"])
        if seg["frames"][-1] > g["last"]:
            g["last"] = seg["frames"][-1]
            g["tail_run"] = {"seg": seg, **seg["runs"][-1]}
        g["links"].append({"a": "洞", "b": seg["tid"], "score": round(sc, 3),
                           "in": d1, "out": d2})
        merged.append(s)
    for s in merged:
        groups.remove(s)


def _finish(g: dict, n_frames: int | None) -> dict:
    tids = sorted(s["tid"] for s in g["segs"])
    frames = set(f for s in g["segs"] for f in s["frames"])
    # 覆盖率用**去重后的帧数**：一条 tid 中间断过、或者两段的检测帧撞车时，
    # 按段累加会把同一帧算两次，覆盖率会超过 100%。
    scores = [l["score"] for l in g["links"]]
    return {"gid": tids[0], "tids": tids,
            "confidence": round(min(scores), 3) if scores else 1.0,
            "runs": [[r["f0"], r["f1"]] for r in sorted(g["runs"], key=lambda r: r["f0"])],
            "n_frames": len(frames),
            "coverage": (round(len(frames) / n_frames, 4) if n_frames else None),
            "links": g["links"]}


# ---------------------------------------------------------------------------
# 合并成一条框轨迹（prep_boxes 用）
# ---------------------------------------------------------------------------
def union_pick(segments: list[dict], tids: list[int]) -> dict | None:
    """把若干段拼成 prep_boxes 要的那一个 pick。

    frames/boxes 取并集后按帧号排序。**组内区间两两不相交**（硬约束保证的），
    所以同一帧不会有两个框；万一有（数据被手工改过），保留面积大的那个——
    宁可要一个"大框"，也不要在同一帧上取平均，那会得到一个谁都不是的框。
    """
    want = set(int(t) for t in tids)
    picked = [s for s in segments if int(s["tid"]) in want]
    if not picked:
        return None
    by_frame: dict[int, list[float]] = {}
    for s in picked:
        for f, b in zip(s["frames"], s["boxes"]):
            b = [float(v) for v in b]
            old = by_frame.get(int(f))
            if old is None or (b[2] - b[0]) * (b[3] - b[1]) > (old[2] - old[0]) * (old[3] - old[1]):
                by_frame[int(f)] = b
    frames = sorted(by_frame)
    boxes = [[round(v, 1) for v in by_frame[f]] for f in frames]
    runs, start = [], frames[0]
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            runs.append([start, frames[i - 1]])
            start = frames[i]
    runs.append([start, frames[-1]])
    return {"tid": min(want), "tids": sorted(want), "frames": frames, "boxes": boxes,
            "runs": runs}


# ---------------------------------------------------------------------------
# people_merge.json（人工确认过的分组）
# ---------------------------------------------------------------------------
def load(path: str | Path) -> dict:
    """读分组决策。

    `exists` 和 `groups` 是两件事，但**对"生效什么"已经没区别了**（两者都表示
    "一组都不并"，见 effective）。分开报是为了另一种用途：`exists=False` 是
    "这个人还没决定过"，`exists=True, groups=[]` 是"他点过取消"。界面上的措辞
    要能分开这两种情形。

      文件不在   -> exists=False, groups=[]  = 还没决定过
      文件在、空 -> exists=True,  groups=[]  = 点过取消/用过之后又撤了
    文件坏了当成"没有决策"（不能因为一个损坏的 json 就把整条流水线打挂）。
    """
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"exists": False, "groups": [], "params": {}}
    except (OSError, ValueError):
        return {"exists": False, "groups": [], "params": {}}
    groups = []
    for g in d.get("groups") or []:
        try:
            tids = sorted(int(t) for t in (g.get("tids") or []))
        except (TypeError, ValueError):
            continue
        if len(tids) >= 2:
            groups.append({"gid": tids[0], "tids": tids})
    return {"exists": True, "groups": groups, "params": d.get("params") or {}}


def effective(saved: dict) -> list[dict]:
    """这条素材**当前生效的**分组——**归并默认关闭，要人点了才算**。

    算法只给建议（见 suggest），建议**不自动生效**。一个人在界面上看到的
    永远是他确认过的那几组，没有确认过就是原始的逐条轨迹。

    ## 为什么改成默认关（原来是默认开）

    建议是按**时间连续性 + 体格（框高）**算的，**不用任何外观特征**——所以
    "体格接近的两个人交替出现"在几何上就是不可分的，而实测这条双人素材正好是
    这种（两人身高相近）。默认套用的两种后果不对称：

      · 该并没并  -> 列表上多几行碎片，用户看一眼就知道要点"用这组"
      · 不该并却并 -> **整条脚本用错人**，而它在界面上长得和"并对了"一模一样，
                     要等脚本跑完看动作才发现

    前者是麻烦，后者是错的。所以默认取"不并"，把那一票留给看得到画面的人。

    文件不在（没决策）和文件在但空（明确不并）在这里是同一件事——都不并。
    `load()` 仍然分开报 `exists`，那是给"这个文件是不是用户写的"用的，
    与"生效什么"无关了。
    """
    return list(saved.get("groups") or [])


def save(path: str | Path, groups: list[dict], params: dict | None = None) -> Path:
    """写分组决策。**只写「人」的分组，单段不写**——单段就是不需要合。

    **一组不剩时也要写文件**（写成空的）：空文件的意思是"明确关掉归并"，而
    文件不在的意思是"还没有决策、按建议来"（见 effective）。把空的情况删掉文件，
    就等于"关掉自动归并"这个操作根本表达不出来——下次一读又变回建议了。
    """
    out = {"version": 1,
           "groups": [{"gid": min(int(t) for t in g["tids"]),
                       "tids": sorted(int(t) for t in g["tids"])}
                      for g in groups if len(g.get("tids") or []) >= 2],
           "params": params or {}}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def main() -> int:
    """命令行：对一条素材跑一次分组，把建议打出来（不落盘）。"""
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(description="对 bbx_all.json 跑一次人物归并建议")
    ap.add_argument("bbx_all", help="outputs/<slug>/bbx_all.json")
    ap.add_argument("--gap-max", type=int, default=GAP_MAX)
    ap.add_argument("--score-min", type=float, default=SCORE_MIN)
    a = ap.parse_args()
    d = json.loads(Path(a.bbx_all).read_text(encoding="utf-8"))
    gs = suggest(d["segments"], d.get("n_frames"), gap_max=a.gap_max,
                 score_min=a.score_min)
    for g in gs:
        tids = "+".join(str(t) for t in g["tids"])
        cov = f"{g['coverage']:.0%}" if g["coverage"] is not None else "?"
        print(f"  人 {g['gid']:>3}  {len(g['tids'])} 段  覆盖 {cov:>4}  "
              f"置信 {g['confidence']:.2f}  轨迹 {tids}")
    print(f"  ---- {len(d['segments'])} 段 -> {len(gs)} 组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
