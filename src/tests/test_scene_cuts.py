"""跨切点处理的验证。

    python src/tests/test_scene_cuts.py

为什么单独一个文件：这一段的**每一处都是"写错了照样跑得通"**——不报错、不崩，
只是脚本上多一下或少一下，而那种症状很容易被归因到"模型不准"。开发过程中
在同一个函数里连踩两个坑，都是靠把物理值打出来才看见的：

  1. `prev = x[i]`（把输出按到 0）看着比"推状态"干净，实测反而更差——
     那一刻在做大幅度动作的话，落差就等于动作本身的幅度。
  2. `prev = x[i] - y[i-1]` 少了 `/a`，实际得到 `y[i] = a·y[i-1]`，还留 10% 落差。
  3. **掩蔽写在高通之前**：桥把台阶摊成斜坡，高通照样打出去（漏 44%），
     而且顺手把去阶跃的好处抵消掉。

要盯住的不变量：
  1. 给了 cuts 且开关都开时，切点处的落差必须远小于不处理            —— 第 1/2 组
  2. `--cut-reset` 的落差**恒为 0**（不是"变小"，是恒等）—— 第 1 组
  3. 不给 cuts / 开关全关时，输出与"根本没有这套功能"逐字节相同      —— 第 3 组
  4. 段内的数据不许被动到（去阶跃只改状态，不改段内序列）            —— 第 4 组
     掩蔽窗**只盖切点之后**，切点前的样本逐字节不变                —— 第 2/4 组
  5. 清单读入要挡越界帧号，长度对不上不许硬套                        —— 第 5 组
  6. 检测器的三道闸门各自真的在挡东西                                —— 第 6 组
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import scene_cuts  # noqa: E402
from to_funscript import blank_cuts, cut_jump, cut_steps, highpass  # noqa: E402

FPS = 30.0
FC = 0.5
fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def blanked(x, cuts, half):
    """blank_cuts 是**原地改**、返回的是被抹掉的样本数，包一层方便写断言。"""
    y = x.copy()
    blank_cuts(y, cuts, half)
    return y


def signal_with_step(n=600, cut=300, step=0.25, amp=0.02, seed=0):
    """慢动作 + 一个跨切台阶。慢动作是**故意**的：高频正弦会被高通放行，
    那样测出来的落差里混着真实信号，判据就糊了。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    x = amp * np.sin(2 * np.pi * 0.4 * t) + 0.002 * rng.standard_normal(n)
    x[cut:] += step
    return x, cut


# --------------------------------------------------------------------------
print("\n[1] 去阶跃：切点处的落差恒为 0")
x, cut = signal_with_step()
no_cut = highpass(x, FPS, FC)
with_cut = highpass(x, FPS, FC, [cut])
check("不处理时落差接近台阶本身（0.9×a 那一下）",
      cut_jump(no_cut, [cut]) > 0.5 * 0.25, f"{cut_jump(no_cut, [cut]):.4f}")
check("去阶跃后落差 == 0（恒等，不是变小）",
      cut_jump(with_cut, [cut]) < 1e-12, f"{cut_jump(with_cut, [cut]):.3e}")

# 换个时刻、换个幅度，别只在一种情形上成立
for c, s in ((120, 0.5), (450, 0.05), (301, -0.3)):
    x2, _ = signal_with_step(cut=c, step=s)
    j = cut_jump(highpass(x2, FPS, FC, [c]), [c])
    check(f"  切点{c} 台阶{s:+.2f} -> 落差 {j:.2e}", j < 1e-12)

# --------------------------------------------------------------------------
print("\n[2] 掩蔽：判据是**切点之后的偏离**，不是那一帧的落差")
# 落差这一项被去阶跃压平了：两种顺序都是 0，分不出好坏。
# 真正的差别在切点后那一窗里**偏离了多少**——掩蔽写在高通之前的话，原始台阶
# 被摊成一个 2W 秒的斜坡交给高通，斜坡的低频部分照样漏得出去。
x, cut = signal_with_step()
half = 6
w = 15                                                  # 切点之后看 0.5s
ac = 0.02                                               # 这段真实动作的幅度

def dev(y):
    return float(np.abs(y[cut:cut + w]).max())

plain = highpass(x, FPS, FC)
step_only = highpass(x, FPS, FC, [cut])
after = blanked(step_only, [cut], half)                 # 去阶跃 + 掩蔽（正确顺序）
before = highpass(blanked(x, [cut], half), FPS, FC, [cut])   # 掩蔽在前（错的顺序）

check("不处理时切点后的偏离远大于真实动作", dev(plain) > 5 * ac, f"{dev(plain):.4f}")
check("去阶跃之后偏离就回到真实动作量级", dev(step_only) < 2 * ac, f"{dev(step_only):.4f}")
check("去阶跃+掩蔽 也不超过真实动作量级", dev(after) < 2 * ac, f"{dev(after):.4f}")
check("掩蔽在前会漏出斜坡的低频（这正是当初踩的坑）",
      dev(before) > 2 * dev(after), f"掩蔽在前 {dev(before):.4f} vs 在后 {dev(after):.4f}")
check("两种顺序的落差都远小于不处理（所以落差分不出顺序）",
      max(cut_jump(after, [cut]), cut_jump(before, [cut])) < 0.1 * cut_jump(plain, [cut]),
      f"在后 {cut_jump(after, [cut]):.4f} / 在前 {cut_jump(before, [cut]):.4f}"
      f" / 不处理 {cut_jump(plain, [cut]):.4f}")
check("掩蔽把它自己那一窗真的改动了",
      not np.allclose(after[cut:cut + half + 1], step_only[cut:cut + half + 1]))
# 窗口**只在切点之后**：切点前的样本一个字节都不许动，且插值从切点前那一帧接出去
check("掩蔽不碰切点之前的样本",
      np.array_equal(after[:cut], step_only[:cut]))
check("窗口右端之外不动",
      np.array_equal(after[cut + half + 1:], step_only[cut + half + 1:]))
# 左端锚点**就是切点前那一帧**（单侧窗口才有这个性质），所以插值从那里接出去，
# 切点处的落差被一并抹掉。拿没做去阶跃的原始高通来量才看得出这一步。
raw_after = blanked(plain, [cut], half)
j_raw = abs(float(plain[cut] - plain[cut - 1]))
j_masked = abs(float(raw_after[cut] - raw_after[cut - 1]))
check("落差被摊到整窗上，每帧只剩约 1/(half+2)（左锚点是切点前那一帧）",
      j_masked < j_raw / (half + 1), f"掩蔽后 {j_masked:.5f} vs 原始 {j_raw:.4f}")
check("掩蔽本身不改切点之前的样本", np.array_equal(raw_after[:cut], plain[:cut]))

# --------------------------------------------------------------------------
print("\n[3] 没有 cuts 时，输出与「没有这套功能」逐字节相同")
x, cut = signal_with_step()
check("highpass 不给 cuts == 原来的公式",
      np.array_equal(highpass(x, FPS, FC), highpass(x, FPS, FC, None)))
check("cuts=[] 与 cuts=None 等价",
      np.array_equal(highpass(x, FPS, FC, []), highpass(x, FPS, FC, None)))
check("blank_cuts 没有 cuts 时一个字节都不动",
      np.array_equal(blanked(x, [], half), x))
check("blank_cuts half<=0 时不动", np.array_equal(blanked(x, [cut], 0), x))
check("cut_jump 没有 cuts 时是 0", cut_jump(x, []) == 0.0)

# 参考实现：把 fc<=0 的短路也验一遍（改了签名之后这条容易断）
check("fc<=0 直接返回原数组（仍绕过 cuts 分支）",
      np.array_equal(highpass(x, FPS, 0, [cut]), x))

# --------------------------------------------------------------------------
print("\n[4] 段内数据不许被动到")
x, cut = signal_with_step()
y = highpass(x, FPS, FC, [cut])
# 去阶跃只该影响切点**之后**：切点前的输出与无 cuts 完全一致
check("切点之前的样点逐字节不变",
      np.array_equal(y[:cut], highpass(x, FPS, FC)[:cut]))
# 掩蔽窗之外也不该动
z = y.copy()
n_blanked = blank_cuts(z, [cut], half)
far = np.ones(len(x), bool)
far[cut:cut + half + 1] = False
check("掩蔽窗之外逐字节不变", np.array_equal(z[far], y[far]))
check("掩蔽窗之内确实线性（二阶差分为 0）",
      float(np.abs(np.diff(z[cut:cut + half + 1], 2)).max()) < 1e-9)
# 单侧窗口的帧数：含切点那一帧，所以是 half+1 而不是 2*half+1
check("被抹掉的是 half+1 帧（单侧含切点）", n_blanked == half + 1, str(n_blanked))

# --------------------------------------------------------------------------
print("\n[5] 清单读入")
tmp = Path(tempfile.mkdtemp())
p = tmp / "cuts.json"
scene_cuts.save({"version": 1, "cuts": [5, 100, 999], "n_frames": 200, "fps": 30.0}, p)
check("越界帧号被丢掉", scene_cuts.cut_frames(p, 200) == [5, 100], str(scene_cuts.cut_frames(p, 200)))
check("帧号 0 被丢掉（切到第 0 帧不是切点）",
      scene_cuts.cut_frames(p, 1000) == [5, 100, 999])
check("读不到的路径返回空清单", scene_cuts.cut_frames(tmp / "nope.json") == [])
check("半截 JSON 返回空清单，不抛",
      (p.with_name("bad.json").write_text("{oops", encoding="utf-8") or True)
      and scene_cuts.cut_frames(p.with_name("bad.json")) == [])
check("cut_steps 给出带符号的台阶",
      cut_steps(np.array([0.0, 1.0, 4.0]), [1]) == [1.0])

# --------------------------------------------------------------------------
print("\n[6] 检测器的三道闸门")
H, S, P, G, F, SC = (scene_cuts.C_HIST, scene_cuts.C_SUST, scene_cuts.C_PROM,
                     scene_cuts.C_GRAY, scene_cuts.C_FRAC, scene_cuts.C_SCORE)


def row(hist=0.0, sust=0.0, prom=0.0, gray=0.0, frac=0.0):
    r = np.zeros(6)
    r[H], r[S], r[P], r[G], r[F] = hist, sust, prom, gray, frac
    r[SC] = hist + gray
    return r


s = np.zeros((40, 6))
s[20] = row(0.5, 0.5, 12.0, 0.2, 0.4)     # 三项全过 -> 应当选中
s[30] = row(0.5, 0.5, 1.5, 0.2, 0.4)      # 突出度不够（渐变）-> 挡
s[10] = row(0.5, 0.05, 12.0, 0.2, 0.4)    # 持续差异不够（闪光）-> 挡
cuts, info = scene_cuts.pick_cuts(s, FPS, abs_min=0.2, k=6.0, min_gap=0.1)
check("三项都过才入选", cuts == [20], str(cuts))
check("阈值来自中位数+MAD（不写死）", info["thr"] == 0.2, str(info["thr"]))
check("帧数太少不报切点", scene_cuts.pick_cuts(np.zeros((1, 6)), FPS)[0] == [])
# min_gap：一次转场里只留最强的那一帧
s3 = np.zeros((40, 6))
s3[20], s3[22] = row(0.4, 0.4, 9.0, 0.2, 0.3), row(0.6, 0.6, 9.0, 0.3, 0.4)
s3[35] = row(0.5, 0.5, 9.0, 0.2, 0.3)
c3, _ = scene_cuts.pick_cuts(s3, FPS, abs_min=0.2, k=6.0, min_gap=0.4)
check("转场邻帧只留最强的那个", c3 == [22, 35], str(c3))

# 突出度本身：单帧尖峰 vs 连续多帧的渐变。这两个是实测里最难分的一对
# （远近镜头切换的 d_gray 只有 0.06，和淡入一个量级；能分开的只有"孤立性"）。
hists = np.tile(np.eye(scene_cuts.BINS, dtype=np.float32)[:1], (120, 1))
hists = np.repeat(np.eye(scene_cuts.BINS, dtype=np.float32)[[0]], 120, axis=0)
hists[60] = np.eye(scene_cuts.BINS, dtype=np.float32)[10]          # 单帧整体换箱
sc1 = scene_cuts.scores_from(hists, np.zeros(120), np.zeros(120), FPS)
check("单帧整体换箱 -> 突出度很高", sc1[60, P] > scene_cuts.PROM_MIN,
      f"{sc1[60, P]:.1f}x")
hists2 = np.repeat(np.eye(scene_cuts.BINS, dtype=np.float32)[[0]], 120, axis=0)
for k in range(50, 70):                                            # 连续 20 帧慢慢换
    hists2[k] = np.eye(scene_cuts.BINS, dtype=np.float32)[k - 50]
sc2 = scene_cuts.scores_from(hists2, np.zeros(120), np.zeros(120), FPS)
check("连续 20 帧渐变 -> 突出度上不去",
      sc2[50:70, P].max() < scene_cuts.PROM_MIN, f"{sc2[50:70, P].max():.1f}x")
check("  渐变里 d_sust 依然很高（所以只靠它挡不住）",
      sc2[60, S] > scene_cuts.SUST_MIN, f"{sc2[60, S]:.2f}")

# --------------------------------------------------------------------------
print("\n[7] 判据可调：默认值只有一份、越界值夹回来、旋钮真的有效")
check("defaults() 就是 TUNABLE 那一份", scene_cuts.defaults() == scene_cuts.TUNABLE)
check("LIMITS 覆盖每一个可调项（漏一个面板上那个滑杆就没有范围）",
      set(scene_cuts.LIMITS) == set(scene_cuts.TUNABLE))
n = scene_cuts.normalize({"prom_min": 9999, "min_gap": -1, "abs_min": "abc", "不认识": 1})
check("超上界夹到上界", n["prom_min"] == 50.0, str(n["prom_min"]))
check("负数夹到下界", n["min_gap"] == 0.0, str(n["min_gap"]))
check("非数字忽略、退回默认", n["abs_min"] == scene_cuts.ABS_MIN, str(n["abs_min"]))
check("不认识的键被丢掉", "不认识" not in n)
check("change_level 保持整数（它是灰度级数）", isinstance(n["change_level"], int))
check("缺的键补默认值", scene_cuts.normalize({}) == scene_cuts.TUNABLE)
check("传 None 也不炸", scene_cuts.normalize(None) == scene_cuts.TUNABLE)

# 旋钮必须真的接在判据上——只改默认值不接到 pick_cuts，面板就是摆设
s7 = np.zeros((40, 6))
s7[20] = row(0.5, 0.5, 12.0, 0.2, 0.4)
s7[30] = row(0.5, 0.5, 6.0, 0.2, 0.4)
check("突出度调到 10 会挡掉 6x 的那个候选",
      scene_cuts.pick_cuts(s7, FPS, abs_min=0.2, k=6, min_gap=0.1, prom_min=10)[0] == [20])
check("调回 5 它又回来",
      scene_cuts.pick_cuts(s7, FPS, abs_min=0.2, k=6, min_gap=0.1, prom_min=5)[0] == [20, 30])
check("最小间隔调大能合并相邻两个",
      len(scene_cuts.pick_cuts(s7, FPS, abs_min=0.2, k=6, min_gap=1.0, prom_min=5)[0]) == 1)

print(f"\n===== {len(passes)} 通过, {len(fails)} 失败 =====")
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
