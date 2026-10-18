"""「多人轨迹」这一步的两个闸。

    python src/tests/test_track_gate.py

为什么要有闸，用一个真实素材说：BV1A（IVE 的 HEYA，4K60）6727 帧里有 **110 个
切点**——平均每 1.1 秒切一次镜头。每切一次 ByteTrack 就丢人重捕，于是跟出的
"轨迹"全是碎片：21 条 ≥3%，最高的才 6%，另有 426 条碎片被当噪声丢掉。而每条
轨迹都要走一遍完整流水线，prep_boxes 会把 bbx 补成全长、对**全部 6727 帧**提
HMR2 特征——实测每条 ≈20 分钟、≈1.8 GB（其中 372 MB 是整片视频的副本）。
21 条就是六七个小时、几十 GB，而且产物是"在两次不相关镜头之间线性插值的框"。

对照：BV1RZY4zPEw2（双人）是 100% / 61% / 33% 三条——那才是这个功能要服务的
场景（一个人被跟踪切成了几段）。所以判据是**占比**，不是"有几条"。

这里把 run_step 打桩，只验"决定跑哪几条"，不真跑任何推理。
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import batch_run  # noqa: E402

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


_NEXT = [0]


def seg(tid: int, cov: float, runs: int = 1) -> dict:
    """一条轨迹段。**段与段在时间上错开**（真实的碎片是先后出现的）——
    原来所有段都挤在第 0 帧、框也一样，那在归并看来就是同一个人被切了很多段，
    会全并成一组，于是这些用例测的就不是它们想测的东西了。"""
    n = 5 * max(1, runs)
    start = _NEXT[0]
    _NEXT[0] = start + n + 30
    frames = list(range(start, start + n))
    return {"tid": tid, "coverage": cov, "runs": [[start, start + n - 1]] * runs,
            "frames": frames, "boxes": [[100.0, 200.0, 200.0, 340.0]] * n,
            "area_sum": cov * 100}


def run_gate(segs: list[dict], opts: dict | None = None, merge="off",
             steps=("preprocess", "boxes", "rtmpose", "gvhmr", "export", "tracks")):
    """把 _run_extra_tracks 跑一遍，返回 (被调用的命令列表, 日志)。"""
    tmp = Path(tempfile.mkdtemp(prefix="gate_"))
    try:
        rt = tmp / "gvhmr_rtmpose" / "X_1280p_30fps"
        rt.mkdir(parents=True)
        (rt / "bbx_all.json").write_text(json.dumps(
            {"n_frames": 100, "segments": segs}), encoding="utf-8")
        out = tmp / "outputs"
        if merge == "off":
            # **关掉自动归并**（写一个空的决策文件）。这几个用例测的是"占比阈值
            # 和条数上限"两个闸，而归并默认开着——不关的话碎片会被并成组，
            # 于是这一段就没有"碎片"可筛了。
            out.mkdir(parents=True, exist_ok=True)
            (out / "people_merge.json").write_text('{"version":1,"groups":[]}',
                                                   encoding="utf-8")
        elif merge:
            out.mkdir(parents=True, exist_ok=True)
            (out / "people_merge.json").write_text(
                json.dumps({"version": 1, "groups": merge}), encoding="utf-8")
        ctx = {"out_root": out, "rt_root": tmp / "gvhmr_rtmpose",
               "rt_dir": rt, "proc_video": tmp / "X_1280p_30fps.mp4",
               "steps": list(steps), "step_opts": {"tracks": opts or {}}}
        calls: list[list[str]] = []
        real = batch_run.run_step
        batch_run.run_step = lambda cmd, desc, index, on_prog, **kw: calls.append(list(cmd))
        b = io.StringIO()
        try:
            with contextlib.redirect_stdout(b):
                batch_run._run_extra_tracks(ctx, lambda f: None)
        finally:
            batch_run.run_step = real
        return calls, b.getvalue(), ctx
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tids_of(calls: list[list[str]]) -> list[int]:
    """从命令行里把 --track <tid> 抠出来。"""
    out = []
    for c in calls:
        if "--track" in c:
            out.append(int(c[c.index("--track") + 1]))
    return out


# --------------------------------------------------------------------------
print("[1] BV1A 那种素材：一条都不该跑（碎片不是人）")
segs = [seg(578, 0.042)] + [seg(t, 0.03 + 0.005 * (21 - i)) for i, t in enumerate(range(21), 1)]
calls, log, ctx = run_gate(segs)
check(calls == [], f"20 条碎片一条都没跑（实际 {len(calls)} 次 run_step）")
check(ctx.get("extra_tracks") in (None, []), f"extra_tracks 是空的：{ctx.get('extra_tracks')}")
check("阈值" in log and "切镜头" in log, "日志里说清了为什么一条都不跑")
# 报的是"剩下多少条"——归并之后先剔掉已并进组的那几条，池子比"其余"小，
# 所以这句话跟着池子走（原来写死"其余 N 条"，加了归并就对不上了）。
check("20 条" in log or "19 条" in log or "剩下" in log or "其余" in log,
      f"日志报了几条：{log.strip()[-90:]}")

print("\n[2] BV1RZY4zPEw2 那种素材：三条全过（主轨迹外跑 2 条）")
segs = [seg(2, 1.0, 7), seg(1, 0.61, 4), seg(8, 0.33, 1)]
calls, log, ctx = run_gate(segs)
check(tids_of(calls) == [1, 8], f"跑了 track 1 和 8：{tids_of(calls)}")
check(len(calls) == 8, f"每条 4 步（检测框/RTMPose/推理/导出）= 8 次：{len(calls)}")
check(ctx["extra_tracks"] == [1, 8], f"extra_tracks 记下了：{ctx['extra_tracks']}")

print("\n[3] 上限：够格的超过 3 条时只跑前 3 条（按占比）")
segs = [seg(2, 1.0)] + [seg(11, 0.70), seg(12, 0.55), seg(13, 0.40),
                        seg(14, 0.35), seg(15, 0.25)]
calls, log, ctx = run_gate(segs)
check(tids_of(calls) == [11, 12, 13], f"只跑占比最高的 3 条：{tids_of(calls)}")
check("2 条" in log and "上限" in log, f"日志说了跳掉几条、为什么：{log.strip()[-80:]}")

print("\n[4] 两个数真的生效（调小了阈值就该跑）")
segs = [seg(578, 0.042)] + [seg(t, 0.03 + 0.005 * (21 - i)) for i, t in enumerate(range(21), 1)]
calls, _, ctx = run_gate(segs, {"min_coverage": 0.05, "max_tracks": 2})
check(tids_of(calls) == [2, 3] or len(tids_of(calls)) == 2,
      f"阈值降到 5%、上限 2 → 跑 2 条：{tids_of(calls)}")
calls, _, _ = run_gate(segs, {"min_coverage": 0.9})
check(calls == [], "阈值抬到 90% → 一条都不跑")
calls, _, _ = run_gate(segs, {"min_coverage": 0.0, "max_tracks": 10})
check(len(tids_of(calls)) == 10, f"上限 10 → 跑 10 条：{len(tids_of(calls))}")

print("\n[5] 参数越界要被夹住，不能把 100 条都放出去")
n = lambda raw: batch_run.normalize_step_opts("tracks", raw)
check(n({"max_tracks": 999})["max_tracks"] == 10, "上限夹到 10")
check(n({"min_coverage": -1})["min_coverage"] == 0.02, "阈值有下限")
check(n({"max_tracks": 0})["max_tracks"] == 1, "至少跑 1 条")
check(n(None) == {"min_coverage": 0.20, "max_tracks": 3, "only": ""},
      "不给就用默认值 20% / 3 条 / 不限定轨迹")
check(n({"only": "  1, 8 ,不是数字 "})["only"] == "1,8,不是数字",
      f"only 只去空白、不夹数值范围：{n({'only': '  1, 8 ,不是数字 '})['only']!r}")

print("\n[5b] 「导出 npz」的坐标系选项")
e = lambda raw: batch_run.normalize_step_opts("export", raw)
check(e(None)["spaces"] == "incam", "默认只导相机系")
check(e({"spaces": "incam,global"})["spaces"] == "incam,global", "两个都要认")
check(e({"spaces": "global"})["spaces"] == "global", "只要世界系也认")
check(e({"spaces": "global-ish"})["spaces"] == "incam", "拼错的值落回默认，不猜")
check(e({"spaces": ""})["spaces"] == "incam", "空值落回默认")
check(batch_run.step_opts_default("export") == {"spaces": "incam"}, "默认值表就这一条")

print("\n[6] 边上几种情况")
calls, log, _ = run_gate([seg(2, 1.0)])
check(calls == [] and "只跟出一条" in log, "只有一条轨迹：不跑，且说明")
calls, log, _ = run_gate([seg(2, 1.0), seg(1, 0.5)], steps=("tracks",))
check(len(calls) == 4, f"单独补跑这一步（只勾 tracks）也要走完 4 步：{len(calls)}")
check(any("export_gvhmr" in " ".join(c) for c in calls), "单独补跑时导出也要跑")

print()
if FAILS:
    print(f"✗ {len(FAILS)} 项没过：")
    for f in FAILS:
        print("   -", f)
    raise SystemExit(1)
print("✓ 全部通过")
