"""把同一个人的多条碎轨迹归并成一条：分组规则的单测。

    python src/tests/test_tracklet_merge.py

为什么要有这个模块：ByteTrack 是在线跟踪，逐帧做 IoU 匹配。丢几帧（快动作、
遮挡、切镜头）轨迹就断了，重新检出时是一个新 tid。实测那条双人素材两个人
各被切成 9~12 段。所以**一个人 ≠ 一条轨迹，一个人是一组轨迹段**。

这里盯住的是三件事：

  1. **时间上重叠的两段绝不同组**。这是免费的硬约束——同一时刻只能在一个
     地方。它挡掉的错误配对最多，而且**违反它一定是 bug**（所以可以自动断言，
     不需要人工标注）。
  2. 同一段合成数据里，换号的三段必须并成一组（正例）。
  3. 该分开的必须分开：体格不同、时间隔太久、位置跳太远的都不许并（反例）。
     合并错了的代价比不合并大——动作直接变成另一个人的。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from tracklet_merge import (  # noqa: E402
    GAP_MAX, OVERLAP_MAX, effective, load, save, suggest, union_pick,
)

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def seg(tid: int, rows: list[tuple[int, float, float, float]]) -> dict:
    """rows = [(帧号, 框心x, 框心y, 框高)]，宽固定 = 框高 * 0.5。

    只造 bbx_all.json 里那四个字段（frames/boxes/runs/coverage）——分组只看
    检测框，不看图像。
    """
    frames, boxes = [], []
    for f, cx, cy, h in rows:
        w = h * 0.5
        frames.append(f)
        boxes.append([round(cx - w / 2, 1), round(cy - h / 2, 1),
                      round(cx + w / 2, 1), round(cy + h / 2, 1)])
    runs, start = [], frames[0]
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            runs.append([start, frames[i - 1]])
            start = frames[i]
    runs.append([start, frames[-1]])
    cover = len(frames) / 1000
    return {"tid": tid, "frames": frames, "boxes": boxes, "runs": runs,
            "coverage": round(cover, 4), "area_sum": round(cover * 0.01, 4)}


def walk(tid: int, f0: int, n: int, cx0: float, cy: float, h: float,
         vx: float = 0.0) -> dict:
    """一个人匀速走/扭的一段。"""
    return seg(tid, [(f0 + i, cx0 + vx * i, cy, h) for i in range(n)])


def groups_of(segs: list[dict]) -> list[list[int]]:
    out = []
    for g in suggest(segs, n_frames=1000):
        out.append(sorted(g["tids"]))
    return sorted(out)


def no_overlap_violation(segs: list[dict], groups) -> list[str]:
    """硬约束断言：同一组里任意两段的"同时在场"不许超过 OVERLAP_MAX 帧。"""
    by_tid = {s["tid"]: s for s in segs}
    bad = []
    for g in groups:
        tids = g["tids"]
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = by_tid[tids[i]], by_tid[tids[j]]
                for ra in a["runs"]:
                    for rb in b["runs"]:
                        ov = min(ra[1], rb[1]) - max(ra[0], rb[0]) + 1
                        if ov > OVERLAP_MAX:
                            bad.append(f"{tids[i]} 与 {tids[j]} 重叠 {ov} 帧 {ra}{rb}")
    return bad


def _run() -> None:
    print("\n[1] 正例：同一个人换号三次 → 并成一组")
    # 一个人从 x=100 走到 x=300，中途丢了两次号，每次断 ~10 帧
    segs = [walk(1, 0, 100, 100, 200, 120, vx=1.0),
            walk(8, 110, 100, 210, 200, 120, vx=1.0),
            walk(15, 220, 100, 320, 200, 120, vx=1.0)]
    g = groups_of(segs)
    check(g == [[1, 8, 15]], f"三段并成一组（实际 {g}）")
    check(no_overlap_violation(segs, suggest(segs, n_frames=1000)) == [],
          "没有违反硬约束")

    print("\n[2] 反例：两个人同时出现（区间相交）→ 绝不并")
    segs = [walk(1, 0, 300, 100, 200, 120), walk(2, 100, 300, 400, 210, 125)]
    g = groups_of(segs)
    check(g == [[1], [2]], f"两个同框的人分开（实际 {g}）")

    print("\n[3] 反例：两人交替出现，但体格差一倍 → 不许并")
    # 出场时间不重叠（交替），位置也接得上，唯一的区别是框高 120 vs 240
    segs = [walk(1, 0, 80, 100, 200, 120), walk(2, 90, 80, 105, 260, 240)]
    g = groups_of(segs)
    check(g == [[1], [2]], f"框高差一倍不算同一个人（实际 {g}）")

    print("\n[4] 间隔：超过硬上限才不并（间隔是罚分，不是 45 帧一刀切）")
    segs = [walk(1, 0, 60, 100, 200, 120),
            walk(2, 60 + GAP_MAX + 60, 60, 100, 200, 120)]
    g = groups_of(segs)
    check(g == [[1], [2]], f"隔 {GAP_MAX + 60} 帧（超上限）不并（实际 {g}）")

    print("\n[4b] 真素材的真值：隔 144 帧、几何连续 → 必须并")
    # BV1RZY4zPEw2：甲 100%（tid 2），乙被切成 61%（tid 1）+ 33%（tid 8），
    # 乙中间断了 144 帧。正确答案是 2 组——这就是为什么间隔不能是硬闸。
    segs = [walk(1, 0, 60, 100, 200, 120), walk(8, 60 + 144, 60, 100, 200, 120)]
    g = groups_of(segs)
    check(g == [[1, 8]], f"隔 144 帧仍然并（实际 {g}）")

    print("\n[4c] 同一个人身上开了两条轨（小重叠）→ 并")
    # 实测 tid 23 还在跑的时候 tid 27 已经在同一个人身上起来了，重叠 4 帧
    segs = [walk(1, 0, 60, 100, 200, 120), walk(2, 56, 60, 100, 200, 120)]
    g = groups_of(segs)
    check(g == [[1, 2]], f"重叠 {4} 帧（≤ {OVERLAP_MAX}）算同一个人（实际 {g}）")

    print("\n[5] 反例：位置跳太远 → 不许并")
    # 隔 15 帧，但横跨 3 个框高——外推对不上
    segs = [walk(1, 0, 60, 100, 200, 120), walk(2, 75, 60, 100 + 3 * 120, 200, 120)]
    g = groups_of(segs)
    check(g == [[1], [2]], f"位移 3 倍框高不并（实际 {g}）")

    print("\n[6] 边界：只有一段 → 一组，组内就是它自己")
    segs = [walk(1, 0, 200, 100, 200, 120)]
    g = groups_of(segs)
    check(g == [[1]], f"单段自成一组（实际 {g}）")

    print("\n[7] 顺序无关：乱序输入结果一致")
    segs = [walk(1, 0, 100, 100, 200, 120, vx=1.0),
            walk(8, 110, 100, 210, 200, 120, vx=1.0),
            walk(15, 220, 100, 320, 200, 120, vx=1.0)]
    a = groups_of(segs)
    b = groups_of(list(reversed(segs)))
    check(a == b, f"输入顺序不影响分组（{a} vs {b}）")

    print("\n[8] 补洞：丢号后新号先出现、老号又续上 → 中间那段并进来")
    # 一个人从 x=100 匀速走到 x=200。号 1 在 0~40 和 60~100 各出现一次（同一个号
    # 中间断了），而断口 45~54 里是号 2 —— 这就是 ByteTrack 最典型的错误形态。
    # 前向链看不到"插进中间"，要靠补洞那段接上。
    def walker(tid, frames):
        return seg(tid, [(f, 100.0 + f, 200.0, 120.0) for f in frames])
    s1 = walker(1, list(range(0, 41)) + list(range(60, 101)))
    s2 = walker(2, list(range(45, 55)))
    gs = suggest([s1, s2], n_frames=1000)
    check(groups_of([s1, s2]) == [[1, 2]], f"断口里的号 2 被并进来（实际 {groups_of([s1, s2])}）")
    check(no_overlap_violation([s1, s2], gs) == [], "并完仍然没有区间相交")

    print("\n[8b] 补洞也要对得上：中间那段位置差太远就不并")
    s3 = seg(2, [(f, 100.0 + f + 3 * 120, 200.0, 120.0) for f in range(45, 55)])
    check(groups_of([s1, s3]) == [[1], [2]],
          f"落点差 3 个框高时不许插进去（实际 {groups_of([s1, s3])}）")

    print("\n[8c] 补洞的体格闸同样生效")
    s4 = seg(2, [(f, 100.0 + f, 200.0, 240.0) for f in range(45, 55)])
    check(groups_of([s1, s4]) == [[1], [2]],
          f"框高差一倍不许插进去（实际 {groups_of([s1, s4])}）")

    print("\n[9] 分组结果自带置信度，供界面显示")
    gs = suggest([walk(1, 0, 100, 100, 200, 120, vx=1.0),
                  walk(8, 110, 100, 210, 200, 120, vx=1.0)], n_frames=1000)
    ok = all("confidence" in g and 0.0 <= g["confidence"] <= 1.0 for g in gs)
    check(ok, "每个组都有 0~1 的 confidence")
    check(all("gid" in g for g in gs), "每个组都有 gid（= 组内最小 tid）")


def test_union_pick() -> None:
    """并集拼出来的那一条框轨迹（prep_boxes --merge-tids 用的就是它）。"""
    print("\n[10] union_pick：把几段拼成一条稠密框轨迹")
    a = seg(1, [(0, 100.0, 200.0, 120.0), (1, 101.0, 200.0, 120.0)])
    b = seg(8, [(5, 110.0, 200.0, 120.0), (6, 111.0, 200.0, 120.0)])
    p = union_pick([a, b], [1, 8])
    check(p is not None and p["frames"] == [0, 1, 5, 6], f"帧号取并集并排序：{p and p['frames']}")
    check(p["runs"] == [[0, 1], [5, 6]], f"runs 是并集里的连续段：{p and p['runs']}")
    check(p["tid"] == 1, "编号取组内最小 tid（下游就用它当这条'人'的编号）")

    small = seg(2, [(3, 100.0, 200.0, 60.0)])
    big = seg(9, [(3, 100.0, 200.0, 160.0)])
    p = union_pick([small, big], [2, 9])
    h = p["boxes"][0][3] - p["boxes"][0][1]
    check(abs(h - 160.0) < 1e-6, f"同一帧撞车时保留面积大的那个框（实际高 {h:.0f}）")
    check(union_pick([a, b], [77]) is None, "一条都不在 → None（调用方报错，不许静默）")


def test_save_load(tmp: Path) -> None:
    """分组决策落盘/读回。界面应用一次 -> 归并步骤读它去跑。"""
    print("\n[11] people_merge.json：只写「人」的分组，单段不写")
    f = tmp / "people_merge.json"
    save(f, [{"tids": [8, 2, 15]}, {"tids": [7]}, {"tids": [3, 4]}])
    d = load(f)
    check([g["tids"] for g in d["groups"]] == [[2, 8, 15], [3, 4]],
          f"单段被丢掉、tid 排序、gid = 最小 tid：{d['groups']}")
    check(not (tmp / "不存在.json").exists() and load(tmp / "不存在.json")["groups"] == [],
          "文件不在 = 没有决策（不是错误）")
    f.write_text("{ 坏掉的 json", encoding="utf-8")
    check(load(f)["groups"] == [], "文件坏了也当没有决策，不许把整条流水线打挂")
    # 撤销归并 = 写一个**空的**决策文件。空文件和没有文件在"生效什么"上是同一
    # 件事（都不并），留着这个区别是为了界面知道该说"还没确认过"还是"已取消"。
    save(f, [{"tids": [1]}])                       # 只剩单段 = 等于没分组
    check(f.exists() and load(f)["groups"] == [], "撤销归并要留下空文件（不是删掉）")
    check(load(f)["exists"] is True, "空文件读回来 exists=True = 决定过（已取消）")
    check(load(tmp / "从没写过.json")["exists"] is False, "文件不在 = 还没决定过")

    print("\n[12] effective：归并默认**关闭**——建议不自动生效")
    segs = [
        {"tid": 1, "coverage": 0.5, "frames": list(range(0, 50)), "runs": [[0, 49]],
         "boxes": [[100.0 + i, 200.0, 160.0 + i, 320.0] for i in range(50)]},
        {"tid": 8, "coverage": 0.5, "frames": list(range(60, 110)), "runs": [[60, 109]],
         "boxes": [[160.0 + i, 200.0, 220.0 + i, 320.0] for i in range(50)]},
    ]
    # 这两段几何上就是一个人（体格一样、时间上勉强接得上）——算法会建议并
    f.unlink(missing_ok=True)
    check([g["tids"] for g in suggest(segs, 1000)] == [[1, 8]],
          "建议照样算得出来（它只是不再自动生效）")
    check(effective(load(f)) == [], "没决策文件 -> 一组都不并")
    save(f, [{"tids": [1, 8]}])                    # 人点过"用这组"
    check([g["tids"] for g in effective(load(f))] == [[1, 8]], "确认过 -> 用文件里那几组")
    save(f, [])                                    # 又点了取消
    check(effective(load(f)) == [], "文件在但空 -> 撤销归并，回到原始轨迹")

    print("\n[13] 噪声轨（dropped）也进建议")
    # 3% 以下被丢弃的那条，其实是同一个人在遮挡期间唯一的检出痕迹；
    # 建议要能把它并进来，否则用户按建议合并后会缺那几秒。
    dropped = [{"tid": 9, "coverage": 0.02, "frames": [52, 53, 54], "runs": [[52, 54]],
                "boxes": [[151.0, 200.0, 211.0, 320.0]] * 3}]
    gs = suggest(segs + dropped, 1000)
    check(el := [g["tids"] for g in gs], f"dropped 也能被并进来：{el}")
    check(any(9 in t for t in el), "那条噪声轨出现在某个组里")


def main() -> int:
    _run()
    test_union_pick()
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="mergetest_"))
    try:
        test_save_load(tmp)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
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
