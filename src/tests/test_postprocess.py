"""输出前那两个后处理：高度相关限位 + 位移→旋转补偿。

    python src/tests/test_postprocess.py

它们作用在**归一化之后的 0~1 数组**上（0.5 = 中立），也就是写完就是 funscript
的那一步。所以要盯的是四件事：

  1. **关掉时逐元素不变**。这是回归线：默认全关，任何改动都不该动到已有素材。
  2. A：低于起始高度一动不动；抬到顶按「顶端倍率」收拢；中立位本身不动
     （缩放的是"离中立多远"，不是位置）。
  3. B：方向要对——向右走了就**向左**倾（默认增益为负），大小按增益线性。
  4. B 补偿的是 **A 之后**的行程（机器实际会走多少就补多少），而且补完要夹回 0~1。
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

import numpy as np                     # noqa: E402
from to_funscript import (                      # noqa: E402
    POST_STEPS, parse_post_order, postprocess,
)

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def axes(**kw) -> dict:
    """六条轴，默认全在中立位。"""
    out = {k: np.full(8, 0.5) for k in ("L0", "L1", "L2", "R0", "R1", "R2")}
    for k, v in kw.items():
        out[k] = np.asarray(v, dtype=float)
    return out


def _run() -> None:
    print("\n[1] 关掉时逐元素不变（回归线）")
    a = axes(L0=[0.1, 0.5, 0.9], L1=[0.0, 0.5, 1.0], L2=[0.2, 0.8, 0.5],
             R1=[0.3, 0.5, 0.7], R2=[0.6, 0.4, 0.5])
    got, stats = postprocess(a, list(POST_STEPS), False, 0.5, 0.0, False, -0.5, -0.5)
    check(all(np.array_equal(got[k], a[k]) for k in a), "两个开关都关 -> 一模一样")
    check(stats == {"order": []}, f"关掉时一步都没跑（统计里只有个空顺序）：{stats}")

    print("\n[2] A：起始高度以下一动不动")
    a = axes(L0=[0.0, 0.3, 0.49], L1=[0.1, 0.9, 0.25], L2=[0.9, 0.1, 0.5])
    got, _ = postprocess(a, list(POST_STEPS), True, 0.5, 0.0, False, 0, 0)
    check(np.array_equal(got["L1"], a["L1"]) and np.array_equal(got["L2"], a["L2"]),
          "L0 < 起始高度时 L1/L2 原样")

    print("\n[3] A：抬到顶收到「顶端倍率」")
    # L0 = 1.0（顶），起点 0.5 -> h=1 -> g = top
    a = axes(L0=[1.0], L1=[1.0], L2=[0.0])
    got, st = postprocess(a, list(POST_STEPS), True, 0.5, 0.0, False, 0, 0)
    check(abs(got["L1"][0] - 0.5) < 1e-9 and abs(got["L2"][0] - 0.5) < 1e-9,
          f"top=0 -> 完全居中（L1={got['L1'][0]:.3f} L2={got['L2'][0]:.3f}）")
    got2, _ = postprocess(a, list(POST_STEPS), True, 0.5, 0.25, False, 0, 0)
    check(abs(got2["L1"][0] - (0.5 + 0.5 * 0.25)) < 1e-9,
          f"top=0.25 -> 保留四分之一行程（{got2['L1'][0]:.3f}）")
    check(abs(st["height_limit"]["min_g"] - 0.0) < 1e-9, "统计里报了最紧那个倍数")

    print("\n[4] A：中立位本身不动，缩放的是离中立的距离")
    a = axes(L0=[1.0, 1.0], L1=[0.5, 0.75], L2=[0.5, 0.5])
    got, _ = postprocess(a, list(POST_STEPS), True, 0.5, 0.0, False, 0, 0)
    check(abs(got["L1"][0] - 0.5) < 1e-9, "L1 在中立位 -> 还是中立位")
    check(abs(got["L1"][1] - 0.5) < 1e-9, "L1 偏离 0.25 -> 收到中立（top=0）")

    print("\n[5] A：半途线性（阈值 + 线性）")
    # L0 = 0.75，起点 0.5，top = 0 -> h = 0.5 -> g = 0.5
    a = axes(L0=[0.75], L1=[1.0], L2=[0.0])
    got, _ = postprocess(a, list(POST_STEPS), True, 0.5, 0.0, False, 0, 0)
    check(abs(got["L1"][0] - 0.75) < 1e-9 and abs(got["L2"][0] - 0.25) < 1e-9,
          f"半高时行程减半（L1={got['L1'][0]:.3f} L2={got['L2'][0]:.3f}）")

    print("\n[6] B：方向——向右走了就向左倾（默认负增益）")
    a = axes(L1=[0.5], L2=[1.0], R1=[0.5], R2=[0.5])
    got, st = postprocess(a, list(POST_STEPS), False, 0, 0, True, -0.5, -0.5)
    check(got["R1"][0] < 0.5, f"L2 向右（1.0）-> R1 向左（{got['R1'][0]:.3f}）")
    check(abs(got["R1"][0] - 0.25) < 1e-9, "大小 = 位移 × 增益（0.5 × -0.5）")
    check("R1" in st["compensate"] and st["compensate"]["R1"]["src"] == "L2",
          "统计里说清了是哪条轴补的")
    a2 = axes(L1=[1.0], L2=[0.5], R1=[0.5], R2=[0.5])
    got2, _ = postprocess(a2, list(POST_STEPS), False, 0, 0, True, -0.5, -0.5)
    check(got2["R2"][0] < 0.5, f"L1 向前（1.0）-> R2 向后倾（{got2['R2'][0]:.3f}）")
    check(abs(got2["R1"][0] - 0.5) < 1e-9, "没动的轴不该被碰")

    print("\n[7] B：增益 0 = 单条关掉；正增益 = 同向")
    got, st = postprocess(axes(L2=[1.0]), list(POST_STEPS), False, 0, 0, True, 0.0, 0.0)
    check(abs(got["R1"][0] - 0.5) < 1e-9 and st["compensate"] == {},
          "两条都给 0 -> 什么都不做")
    got2, _ = postprocess(axes(L2=[1.0]), list(POST_STEPS), False, 0, 0, True, 0.0, 0.3)
    check(abs(got2["R1"][0] - 0.65) < 1e-9, f"正增益就同向（{got2['R1'][0]:.3f}）")

    print("\n[8] B：补偿的是 A 之后的行程，补完夹回 0~1")
    # L0 顶 + top=0 -> L2 被收到 0.5；补偿用的是这个 0.5，不是原始的 1.0
    a = axes(L0=[1.0], L2=[1.0], R1=[0.5])
    got, _ = postprocess(a, list(POST_STEPS), True, 0.5, 0.0, True, 0.0, -0.5)
    check(abs(got["R1"][0] - 0.5) < 1e-9,
          f"限位之后位移是 0 -> 补偿也为 0（{got['R1'][0]:.3f}）")
    # R1 本来就在 0.1，再补 -0.25 -> 会到 -0.15，必须夹回 0 以上
    a2 = axes(L2=[1.0], R1=[0.1])
    got2, st2 = postprocess(a2, list(POST_STEPS), False, 0, 0, True, 0.0, -0.5)
    check(0.0 <= got2["R1"][0] <= 1.0, f"补过头要夹住（{got2['R1'][0]:.4f}）")
    check(st2["compensate"]["R1"]["clipped"] > 0,
          f"并在统计里报了削顶比例（{st2['compensate']['R1']['clipped']:.0%}）")


def test_order_and_gain() -> None:
    print()
    print("[9] 逐轴倍乘：只缩放离中立多远")
    a = axes(L0=[1.0, 0.0], L1=[0.5], R1=[0.75])
    got, st = postprocess(a, list(POST_STEPS), False, 0, 0, False, 0.0, 0.0,
                          {"L0": 0.5, "R1": 0.0, "L1": 1.0})
    check(abs(got["L0"][0] - 0.75) < 1e-9 and abs(got["L0"][1] - 0.25) < 1e-9,
          f"L0 ×0.5 -> 行程减半（{got['L0'][0]:.3f} / {got['L0'][1]:.3f}）")
    check(abs(got["R1"][0] - 0.5) < 1e-9, f"×0 -> 锁在中立位（{got['R1'][0]:.3f}）")
    check(got["L1"][0] == a["L1"][0], "×1 的轴逐位不动")
    check(st["gain"]["L0"]["gain"] == 0.5, "统计里报了每条轴的倍率")
    check(st["order"] == ["gain"], f"只跑了 gain 这一步：{st['order']}")

    print()
    print("[10] 顺序是参数：先限位再补偿 ≠ 先补偿再限位")
    a = axes(L0=[1.0], L1=[1.0], L2=[0.5], R2=[0.5], R1=[0.5])
    first, s1 = postprocess(a, ["height_limit", "compensate"], True, 0.5, 0.0,
                            True, -0.5, 0.0)
    second, s2 = postprocess(a, ["compensate", "height_limit"], True, 0.5, 0.0,
                             True, -0.5, 0.0)
    check(s1["order"] == ["height_limit", "compensate"] and
          s2["order"] == ["compensate", "height_limit"], "报告里按实际顺序念")
    check(abs(first["R2"][0] - 0.5) < 1e-9,
          f"先限位（L1 被收到 0.5）再补偿 -> 没得补（{first['R2'][0]:.3f}）")
    check(abs(second["R2"][0] - 0.25) < 1e-9,
          f"先补偿（按原始 L1=1.0 补）再限位 -> 补过了（{second['R2'][0]:.3f}）")
    check(not np.array_equal(first["R2"], second["R2"]), "两种顺序结果不同（不可交换）")

    print()
    print("[11] 顺序串的解析：认不出的丢掉、没提到的补在末尾")
    check(parse_post_order("gain, 拼错的 ,height_limit") ==
          ["gain", "height_limit", "compensate"], "只留认得出的，缺的补在末尾")
    check(parse_post_order("") == list(POST_STEPS), "空串 -> 默认顺序")
    check(parse_post_order("gain,gain") == ["gain", "height_limit", "compensate"],
          "重复的只算一次")
    check(parse_post_order(None) == list(POST_STEPS), "None -> 默认顺序")


def main() -> int:
    _run()
    test_order_and_gain()
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
