"""TCode 时间基的验证。

    python src/tests/test_tcode_timebase.py

为什么单独一个文件：这个 bug **不表现在数值上**——tcode 和 funscript 逐帧
完全一致（相关系数 1.000、最大差 0.005 = funscript 的量化粒度），只在
**时间**上。TCode 的 `I` 只存整数毫秒，30fps 的真实帧间隔是 33.333ms，
文件里写出来是 `I33`；用 `1000/33 = 30.303fps` 反推就快了 1%。
短素材看不出来（9.5s 偏 94ms），长素材上累积到秒级：268s 偏 2.7s，
表现是 SR6 越跑越跑到曲线前面。

要盯住的不变量：
  1. `read_tcode(d, fps=X)` 返回的就是 X —— 调用方知道真实帧率就别让它猜
  2. `read_tcode(d)` 不带 fps 时返回**取整**值（`I33` -> 30），不是 30.303
  3. 用返回的 fps 把时间映射成行号，与 funscript 在同一时刻的值一致（量化内）
  4. 用旧的 `1000/iv` 反推，长素材上明显偏离 —— 这条是判据本身：
     没有它，上面第 3 条什么都证明不了
"""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from server import convert  # noqa: E402

FS_DIR = ROOT / "workspace" / "funscripts"
QUANT = 0.01          # funscript 的 pos 是整数 0..100，量化粒度就是 0.01
TOL = 2.5 * QUANT     # 正确的映射：一帧内的运动 + 量化

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def val(line: str, axis: str) -> float:
    """`L05000` -> 0.5。位数即精度。"""
    m = re.search(axis + r"(\d{1,4})", line)
    return int(m.group(1)) / 10 ** len(m.group(1))


# ---------------------------------------------------------------- 纯单元
print("\n== T1 read_tcode 的帧率来源 ==")
tmp = Path(tempfile.mkdtemp(prefix="tcfps_"))
(tmp / "x.tcode.txt").write_text(
    "L05000 L15000 I33\nL06000 L14000 I33\nL07000 L13000 I33\n", encoding="utf-8")

tc_known = convert.read_tcode(tmp, fps=30.0)
check("显式给 fps 就用它", tc_known["fps"] == 30.0, repr(tc_known["fps"]))
tc_guess = convert.read_tcode(tmp)
check("不给 fps 时取整（I33 -> 30，不是 30.303）", tc_guess["fps"] == 30.0,
      repr(tc_guess["fps"]))
check("interval_ms 原样带出来（前端/调试要看）", tc_guess["interval_ms"] == 33)
# 60fps 的产物写出来是 I17，1000/17 = 58.8 —— 又是个快 1.5% 的坑，
# 所以显式传 fps 那条路必须有，不能只靠取整。
check("60fps 显式传入时不被取整带偏", convert.read_tcode(tmp, fps=60.0)["fps"] == 60.0)
tc17 = Path(tempfile.mkdtemp(prefix="tcfps_"))
(tc17 / "y.tcode.txt").write_text("L05000 I17\n", encoding="utf-8")
check("I17 取整得 59（真值 60）—— 这就是必须显式传 fps 的原因",
      convert.read_tcode(tc17)["fps"] == 59.0, repr(convert.read_tcode(tc17)["fps"]))
# 但 `I33` 本身分不出 29.97 和 30（两个都写 33），而正式产物里**就有**一条
# 29.97fps 的（BV16Vd3BBEZA，原片就是 29.97）。所以要让同目录的 funscript
# 来说话——它是同一次运行写的，`at = round(i*1000/fps_out)`，能精确还原。
tc29 = Path(tempfile.mkdtemp(prefix="tcfps29_"))
(tc29 / "z.tcode.txt").write_text("L05000 I33\nL06000 I33\n", encoding="utf-8")
n29 = 200
at29 = [int(round(i * 1000.0 / 29.97)) for i in range(n29)]
(tc29 / "z.funscript").write_text(json.dumps(
    {"actions": [{"at": a, "pos": 50} for a in at29]}), encoding="utf-8")
got29 = convert.read_tcode(tc29)["fps"]
check("有同目录 funscript 时用它的时间基（I33 也能认出 29.97）",
      abs(got29 - 29.97) < 0.01, repr(got29))
check("显式 fps 仍然优先于 funscript", convert.read_tcode(tc29, fps=30.0)["fps"] == 30.0)

# ---------------------------------------------------------------- 真产物
print("\n== T2 正式产物上，时间 -> 行号 的映射与曲线一致 ==")
clips = []
for tcf in sorted(FS_DIR.glob("*/*.tcode.txt")):
    d = tcf.parent
    main = [p for p in d.glob("*.funscript") if p.name.count(".") == 1]
    if not main:
        continue
    try:
        acts = json.loads(main[0].read_text(encoding="utf-8"))["actions"]
    except Exception:                                     # noqa: BLE001
        continue
    fps_true = (len(acts) - 1) * 1000.0 / acts[-1]["at"]   # 从时间戳反推真值
    clips.append((d.name, tcf, acts, fps_true))

if not clips:
    print(f"  workspace/funscripts 下没有可用的产物，无法运行：{FS_DIR}")
    sys.exit(1)

print(f"    扫到 {len(clips)} 条产物，逐条比对：")
print(f"    {'素材':18s} {'曲线时长':>8s} {'真 fps':>7s} {'读到 fps':>8s} "
      f"{'逐帧最大差':>10s} {'行号偏移(帧)':>12s} {'旧读法行号偏移':>14s}")
worst_val = 0.0
worst_idx_new = 0
worst_idx_old = 0
for slug, tcf, acts, fps_true in clips:
    tc = convert.read_tcode(tcf.parent)
    lines = tc["lines"]
    dur_ms = acts[-1]["at"]
    n = min(len(lines), len(acts))

    # (a) 同一条曲线：第 i 行 vs 第 i 个 action
    dev = max(abs(val(lines[i], "L0") - acts[i]["pos"] / 100.0) for i in range(n))

    # (b) 同一时间轴：把时间映射成行号，看是不是落在曲线自己所在的那一帧上
    #     （曲线"在某一时刻是哪一帧"= 最后一个 at <= t 的 action 的下标）。
    #     这样比就不含"两个相邻帧之间该取哪一帧"的半帧歧义了。
    ats = [a["at"] for a in acts]
    idx_new = idx_old = 0
    # t 单调递增，所以曲线下标也跟着往前走，不用每个采样点重扫（8000 帧 ×
    # 上千个采样点会慢到没法用）
    j = 0
    t = 0
    while t <= dur_ms:
        while j + 1 < len(ats) and ats[j + 1] <= t:
            j += 1
        idx_new = max(idx_new, abs(min(n - 1, int(t / 1000.0 * tc["fps"])) - j))
        idx_old = max(idx_old, abs(min(n - 1, int(t / 1000.0 * (1000.0 / tc["interval_ms"]))) - j))
        t += 250
    worst_val = max(worst_val, dev)
    worst_idx_new = max(worst_idx_new, idx_new)
    worst_idx_old = max(worst_idx_old, idx_old)
    print(f"    {slug:18s} {dur_ms/1000:7.1f}s {fps_true:7.3f} {tc['fps']:8.3f} "
          f"{dev:10.4f} {idx_new:12d} {idx_old:14d}")

check("每条产物的读到帧率都等于从时间戳反推的真值",
      all(abs(convert.read_tcode(tcf.parent)["fps"] - ft) < 0.01
          for _, tcf, _, ft in clips),
      f"共 {len(clips)} 条")
check(f"tcode 与 funscript 是同一条曲线（逐帧最大差 {worst_val:.4f} <= {TOL}）",
      worst_val <= TOL)
# 时间到行号的映射：真 fps 下最多差 1 帧（`at = round(i*1000/fps)` 的取整），
# 旧的 1000/I 读法在长素材上是几十帧。这条才是这个 bug 的判据。
check(f"时间 -> 行号 对齐，最多差 1 帧（实测 {worst_idx_new} 帧）", worst_idx_new <= 1)
check(f"旧读法（1000/I）明显跑偏（实测最差 {worst_idx_old} 帧 > 3）", worst_idx_old > 3)

# ---------------------------------------------------------------- 预览产物
print("\n== T3 /convert 返回的 tcode 用生成时的帧率 ==")
SLUG = "BV1orhK64E2R"
# 别写进用户正在调的 preview：测试只关心帧率，不需要污染那份产物。
PREV = Path(tempfile.mkdtemp(prefix="tcfps_prev_"))


def _tmp_preview(slug: str) -> Path:
    d = PREV / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


_orig_preview = convert.preview_dir
convert.preview_dir = _tmp_preview
try:
    res = convert.convert(SLUG, {})
except Exception as exc:                                  # noqa: BLE001
    convert.preview_dir = _orig_preview
    print(f"    跳过：{exc}")
else:
    if not res.get("ok"):
        check(f"{SLUG} 能重算", False, str(res.get("error")))
    else:
        tc = res["tcode"]
        check("重算返回了 tcode", tc is not None)
        # 产物帧率 = --rate 优先，其次 --fps；两者都没给才是取整回落
        want = res["params"].get("rate") or res["params"].get("fps") \
            or round(1000.0 / tc["interval_ms"])
        check("fps 用的是生成时真正那个（不是 1000/33）",
              tc["fps"] == want, f"tcode fps={tc['fps']} 期望={want}")
        check("默认重算不再走'按 30 假设'——args.fps 已由后端补上",
              res["params"].get("fps") == 30.0, repr(res["params"].get("fps")))
        # 前端 paramDefaults 带着 `fps: null`，键**一直在**。判据若写成
        # `"fps" not in params` 就会永远跳过补 --fps 那一步，于是每次都走
        # "未能读到帧率" 的隐式分支（告警打在 stderr，报告里看不到）。
        r_null = convert.convert(SLUG, {"fps": None})
        check("params 带 fps:null（前端真实形状）时仍补上 --fps",
              r_null.get("ok") and r_null["params"].get("fps") == 30.0,
              repr(r_null.get("params", {}).get("fps")))
        # 换一个帧率生成，返回的 fps 必须跟着变（证明不是硬编码 30）
        r60 = convert.convert(SLUG, {"fps": 60.0})
        if r60.get("ok"):
            check("--fps 60 时返回 60（跟随生成参数）",
                  r60["tcode"]["fps"] == 60.0, repr(r60["tcode"]["fps"]))
            check("60fps 的 I 是 17，反推只会得到 58.8 —— 所以必须显式传",
                  r60["tcode"]["interval_ms"] == 17, str(r60["tcode"]["interval_ms"]))
        else:
            check(f"{SLUG} 能用 fps=60 重算", False, str(r60.get("error")))
    convert.preview_dir = _orig_preview
    import shutil
    shutil.rmtree(PREV, ignore_errors=True)

print(f"\n===== {len(passes)} 通过, {len(fails)} 失败 =====")
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
