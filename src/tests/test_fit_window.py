"""--fit-window（按时间窗归一化）的验证。全部断言，不做人工目测。

    python src/tests/test_fit_window.py

两段：
  · T1–T7 端到端跑 CLI，需要真素材 + 一份"加这个功能之前"的正式产物做回归基准。
  · T8–T11 纯单元，只依赖 to_funscript 里的两个函数，无素材也能跑。
缺基准会直接退出并指出缺哪个文件（不做静默跳过——回归测试跳过就没意义了）。

回归基准用 workspace/funscripts/BV1orhK64E2R/ 里批处理产出的正式产物——那是
"加 --fit-window 之前"的行为，能直接验证重构没有动到默认路径。

要盯住的不变量（改这部分代码时最容易破的就是这几条）：
  1. 默认参数输出与旧产物**逐字节相同**（阶段 2 关着时不能有任何影响）
  2. 阶段 2 与阶段 1 独立：改 --fit 不改变输出（尺度被约掉）
  3. --fit-window >= 素材长度时精确退化为全程
  4. 阶段 2 开启时 --travel-* / --max-deg 不影响输出
  5. 相对增益封顶 3 倍，且"1.00x"确实等于全程尺度
"""
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PY = r"C:\Python312\python.exe"
ROOT = Path(__file__).resolve().parents[2]
SLUG = "BV1orhK64E2R"
NPZ = (ROOT / f"workspace/outputs/{SLUG}/gvhmr_rtmpose"
       / f"{SLUG}_1280p_30fps/hmr4d_results_incam.npz")
OFFICIAL = (ROOT / f"workspace/funscripts/{SLUG}" / f"{SLUG}_1280p_30fps.funscript")
MAIN = f"{SLUG}_1280p_30fps.funscript"
TMP = Path(tempfile.mkdtemp(prefix="fitwin_"))

fails, passes = [], []

# 缺基准就**直接失败**，不做静默跳过：这是回归测试，跳过等于失去意义。
# 要换素材，改上面的 SLUG 即可。
_missing = [p for p in (NPZ, OFFICIAL) if not p.exists()]
if _missing:
    print("缺少回归基准，无法运行：")
    for p in _missing:
        print(f"   {p}")
    sys.exit(1)


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def run(tag, extra, expect_rc=0):
    out = TMP / tag
    cmd = [PY, "src/to_funscript.py", str(NPZ), "--out", str(out), "--fps", "30"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    if r.returncode != expect_rc:
        print(f"    !! {tag} rc={r.returncode}\n{r.stdout[-2500:]}\n{r.stderr[-1500:]}")
    return r, out


def digest(d: Path) -> str:
    return hashlib.sha256((d / MAIN).read_bytes()).hexdigest()[:16]


def all_digest(d: Path):
    """六轴 + tcode 的全部哈希，用来抓只改次要轴的回归。"""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            for p in sorted(d.glob("*")) if p.is_file()}


def by_axis(text, pat):
    """按行首轴名取值。注意 AXIS_DESC['R0'] = "扭转（绕竖直轴 = 扭胯）" **含空格**，
    所以不能用 \\S+ 去吃描述——那样 R0 会被整个跳过，封顶就永远"没测到"。"""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s+([LR]\d)\s", line)
        if not m:
            continue
        g = re.search(pat, line)
        if g:
            out[m.group(1)] = tuple(float(x) for x in g.groups())
    return out


print("\n== T1 回归：默认参数 == 9/23 批处理正式产物 ==")
r0, o0 = run("default", [])
d0 = digest(o0)
check("默认输出与正式产物逐字节一致", d0 == hashlib.sha256(OFFICIAL.read_bytes()).hexdigest()[:16],
      f"{d0} vs {hashlib.sha256(OFFICIAL.read_bytes()).hexdigest()[:16]}")
m = re.search(r"(\d+) 帧 @ ([\d.]+)fps\s+([\d.]+)s", r0.stdout)
NFRAMES, NFPS, NSEC = int(m.group(1)), float(m.group(2)), float(m.group(3))
print(f"    素材：{NFRAMES} 帧 @ {NFPS:g}fps = {NSEC:.1f}s")

print("\n== T2 开窗确实改变了输出 ==")
r5, o5 = run("win5", ["--fit-window", "5"])
check("--fit-window 5 rc=0", r5.returncode == 0)
check("--fit-window 5 与默认不同",
      all_digest(o5) != all_digest(o0))

print("\n== T3 两段独立：阶段 2 不受阶段 1 影响 ==")
# 阶段 2 把 d 除以窗内分位，而阶段 1 只是把 d 整体乘一个常数 ->
# 该常数在两个分位里同时出现，约掉。所以这两个必须逐字节相同。
r_off, o_off = run("win5_fit0", ["--fit-window", "5", "--fit", "0"])
check("fit 0 + 窗 5 与 fit 99 + 窗 5 逐字节一致（尺度被约掉）",
      all_digest(o_off) == all_digest(o5),
      f"{digest(o_off)} vs {digest(o5)}")
check("fit 0 + 窗 5 与 fit 0（无窗）不同 —— 阶段 2 真的在 --fit 关着时生效",
      all_digest(o_off) != all_digest(run("fit0", ["--fit", "0"])[1]))
check("报告里说明了阶段 2 独立于 --fit",
      "惯例分位" in r_off.stdout, repr(r_off.stdout.splitlines()[:1]))

print("\n== T4 窗长 >= 素材长度 精确退化为全程 ==")
r20, o20 = run("win20", ["--fit-window", "20"])
check("--fit-window 20 与默认逐字节一致（退化）", all_digest(o20) == all_digest(o0),
      f"{digest(o20)} vs {d0}")
check("退化时报告有说明", "退化成全程" in r20.stdout)
check("--fit-window 9999 同样退化",
      all_digest(run("win9999", ["--fit-window", "9999"])[1]) == all_digest(o0))
check("--fit 0 + 窗 9999 与 --fit 0 一致",
      all_digest(run("fit0w9999", ["--fit", "0", "--fit-window", "9999"])[1])
      == all_digest(run("fit0", ["--fit", "0"])[1]))

print("\n== T5 阶段 2 开启时行程组失效（无论 --fit 开不开） ==")
a = run("win5_travel", ["--fit-window", "5", "--travel-l0", "0.3",
                        "--travel-l1", "0.02", "--max-deg", "120"])[1]
check("fit 99 + 窗：改 travel/max-deg 输出不变", all_digest(a) == all_digest(o5))
b = run("win5_fit0_travel", ["--fit-window", "5", "--fit", "0",
                             "--travel-l0", "0.3", "--max-deg", "120"])[1]
check("fit 0 + 窗：改 travel/max-deg 输出不变", all_digest(b) == all_digest(o_off))
# 反过来：窗关着、fit 关着时，travel 必须**是**有效的，否则上面两条是空转
c = run("fit0_travel", ["--fit", "0", "--travel-l0", "0.3"])[1]
check("fit 0 + 无窗：改 travel 输出**变化**（对照组）",
      all_digest(c) != all_digest(run("fit0", ["--fit", "0"])[1]))

print("\n== T6 增益封顶 <= 3x ==")
win_gain = by_axis(r5.stdout, r"(\d+\.\d+)-(\d+\.\d+)x")
check("解析到 6 个轴的窗口增益", len(win_gain) == 6, f"{sorted(win_gain)}")
print("    表（--fit-window 5）:")
for ln in r5.stdout.splitlines():
    if re.match(r"^\s+[LR]\d\s", ln):
        print("   " + ln)
worst = max((hi for _, hi in win_gain.values()), default=0.0)
check("每轴相对增益 <= 3x", worst <= 3.0 + 1e-6, f"实测最大 {worst:.3f}x")
# 报告的增益是**相对全程**的，所以应该以 1.00 为中心（有的轴 >1 有的 <1），
# 而不是绝对值那种整体偏大/偏小的分布。这是上面那个 gain/gain_rel 修复的判据。
lo_all = min(lo for lo, _ in win_gain.values())
check("相对增益以 1.00 为中心（有轴 >1 也有轴 <1）",
      lo_all < 1.0 < worst, f"范围 {lo_all:.2f}–{worst:.2f}")
capped = [ln.split()[0] for ln in r5.stdout.splitlines()
          if ln.strip().endswith("*") and re.match(r"^\s+[LR]\d\s", ln)]
# 这条**不**断言必然触顶：9.5s 的素材配 5s 窗，窗里一直有动作，q 掉不到
# q_ref/3 以下，所以封顶不该被触发。封顶本身由 T10 用合成静段验证。
check("该素材上封顶未被误触发", len(capped) == 0, f"触顶轴 {capped}")
print(f"    本素材最小相对增益 {lo_all:.2f}x（即最大抬升 {1/lo_all:.2f} 倍，封顶 3 倍）")

print("\n== T7 分位参数跟着 --fit 走 ==")
r90 = run("win5_pct90", ["--fit-window", "5", "--fit", "90"])[0]
check("--fit 90 时窗口阶段用 90% 分位", "90% 分位" in r90.stdout)
check("--fit 0 时窗口阶段回落到 99%", "99% 分位" in r_off.stdout)

print("\n== T8 window_scale 的数学（与独立实现对比） ==")
sys.path.insert(0, str(ROOT / "src"))
import to_funscript as tf  # noqa: E402

rng = np.random.default_rng(0)
for T, win, pct in [(300, 5.0, 99), (300, 1.0, 90), (97, 0.5, 50)]:
    arr = np.abs(rng.normal(0, 1, T)) * (1 + np.sin(np.arange(T) / 40.0) ** 2)
    got = tf.window_scale(arr, 30.0, win, pct)
    w = max(3, int(round(win * 30.0)) | 1)
    half = w // 2
    ref = np.empty(T)
    for i in range(T):
        lo, hi = max(0, i - half), min(T - 1, i + half)
        seg = arr[lo:hi + 1]
        if i - half < 0:                     # 边缘延拓：不足处用首值补齐
            seg = np.concatenate([np.repeat(arr[0], half - i), seg])
        if i + half > T - 1:
            seg = np.concatenate([seg, np.repeat(arr[-1], i + half - (T - 1))])
        assert len(seg) == w, (len(seg), w, i)
        ref[i] = np.percentile(seg, pct)
    err = float(np.max(np.abs(got - ref)))
    check(f"T={T} win={win}s pct={pct} 与逐帧独立实现一致", err < 1e-9, f"maxerr {err:.2e}")

print("\n== T9 越窗长度退化为常数 ==")
arr = np.abs(rng.normal(0, 1, 100))
g = tf.window_scale(arr, 30.0, 100.0, 99)
check("窗长 > 素材 时输出为常数", float(g.std()) == 0.0)
check("该常数 = 全段分位", abs(float(g[0]) - float(np.percentile(arr, 99))) < 1e-12)

print("\n== T10 静段不会被放大到满行程 ==")
# 前半段大动作、后半段几乎静止。没有封顶的话后半段会被拉到满行程。
sig = np.concatenate([np.sin(np.arange(600) / 8.0) * 1.0, rng.normal(0, 0.004, 600)])
d_out, gmin, gmax, cap = tf.window_normalize(sig.copy(), 30.0, 5.0, 99.0, 39)
check("相对增益上限 <= 3x", gmax <= 3.0 + 1e-9, f"gmax {gmax:.3f}")
check("静止段确实触到了封顶", gmax > 2.9, f"gmax {gmax:.3f}（封顶值 3.0）")
check("确实有帧被托住", cap > 0.01, f"capped {cap:.1%}")
check("静段输出峰值远小于满行程", float(np.abs(d_out[900:]).max()) < 0.35,
      f"静段峰值 {float(np.abs(d_out[900:]).max()):.3f}（行程极限 = 1.0）")
check("大动作段仍然跑满行程", float(np.abs(d_out[:600]).max()) > 0.9,
      f"峰值 {float(np.abs(d_out[:600]).max()):.3f}")

print("\n== T11 幅度恒定时相对增益应为 1.00x（报告的读法） ==")
# 全程振幅不变 -> 窗内分位 == 全段分位 -> 相对增益处处 1.0。
# 这直接验证"1.00x = 与全程归一化给出的尺度相同"这句注释没说错。
flat = np.sin(np.arange(900) / 8.0) * 1.0
_, fmin, fmax, _ = tf.window_normalize(flat.copy(), 30.0, 5.0, 99.0, 39)
# 容差 1%，不是 1e-6：分位数**依赖样本数**。151 个样本的 99 分位（≈第 2 大）
# 略低于 900 个样本的 99 分位（≈第 9 大），于是振幅恒定时窗内尺度略小、
# 相对增益略大于 1。这是有限窗的固有性质，不是缺陷——设计上就接受。
check("恒定振幅下相对增益 ≈ 1.00x（偏差 <1%）",
      abs(fmin - 1.0) < 0.01 and abs(fmax - 1.0) < 0.01,
      f"{fmin:.6f}–{fmax:.6f}（偏 {max(abs(fmin-1), abs(fmax-1))*100:.3f}%）")
# d_out 是"信号 / 它自己的窗内分位"，所以**本来就是无量纲的**：输入放大 12 倍，
# 输出一模一样（不是跟着放大 12 倍）。这正是阶段 2 能对 --fit 和 --travel-* 都
# 无感的原因——它根本不知道 d 是什么量纲。
d_abs = flat * 12.0
out_a = tf.window_normalize(d_abs, 30.0, 5.0, 99.0, 39)
out_b = tf.window_normalize(flat.copy(), 30.0, 5.0, 99.0, 39)
check("输入放大 12 倍，d 输出不变（本来就无量纲）",
      float(np.abs(out_a[0] - out_b[0]).max()) < 1e-9,
      f"maxerr {float(np.abs(out_a[0] - out_b[0]).max()):.2e}")
check("相对增益不随量纲漂", abs(out_a[1] - fmin) < 1e-12 and abs(out_a[2] - fmax) < 1e-12,
      f"{out_a[1]:.6f}–{out_a[2]:.6f}")

print(f"\n===== {len(passes)} 通过, {len(fails)} 失败 =====")
for f in fails:
    print("  FAILED:", f)
# 中间产物在临时目录里，跑完就删——不要往仓库根目录丢东西
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
