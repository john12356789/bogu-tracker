"""管线步骤依赖的验证。

    python src/tests/test_step_deps.py

为什么单独一个文件：这张表是**界面置灰 + 后端 400** 两边的共同依据，
错了不会崩，只会表现为两种很难查的症状之一——
  · 太松：界面放行，跑了几分钟才在"导出 npz"那步报一句文件不存在；
  · 太紧：正常的"素材算过了，只想补个导出"被拦住，用户只能整条重跑。

要盯住的不变量：
  1. STEP_ORDER 本身是合法拓扑序（process_one 是按它顺序执行的）
  2. PREREQS 是**传递闭包**，不是只有直接上一层（置灰时要一次说清楚缺哪些）
  3. 判据是"勾了 或 盘上有产物"，不是"前置必须被勾上" —— 第 4 组
  4. 产物**不能自证前置**：已经导出的 npz 不能拿来证明 GVHMR 跑过 —— 第 5 组
  5. 依赖表只在 batch_run 定义一次，界面/后端拿到的必须是同一份 —— 第 6 组
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from batch_run import (  # noqa: E402
    DEFAULT_STEPS, PREREQS, STEP_DEPS, STEP_FN, STEP_NAME, STEP_ORDER, check_prereqs,
)
from server.jobs import UI_STEPS  # noqa: E402

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def products(*have):
    """造一个 step_products 形状的字典：列出来的算"盘上已有产物"。"""
    return {s: s in have for s in STEP_ORDER}


# 不默认跑、必须手动勾的几步（它们的代价都是"再来一遍完整流水线"）。
# 「主线」= 剩下的那 5 步：从预处理到导出 npz，一条素材算完就有产物。
OPTIONAL = ("merge", "tracks", "render")
MAINLINE = [s for s in STEP_ORDER if s not in OPTIONAL]


# --------------------------------------------------------------------------
print("\n[1] 依赖表自洽")
check("依赖表的键覆盖全部步骤，不多不少", set(STEP_DEPS) == set(STEP_ORDER),
      f"{sorted(STEP_DEPS)} vs {STEP_ORDER}")
check("没有自依赖", all(s not in STEP_DEPS[s] for s in STEP_ORDER))
check("前置都是已知步骤",
      all(p in STEP_DEPS for s in STEP_ORDER for p in STEP_DEPS[s]))
# preprocess 是源头、render 是终点：这两条断了，上面那堆判断就都是空转
check("preprocess 没有前置", STEP_DEPS["preprocess"] == [])
check("除 preprocess 外每一步都直接或间接依赖它",
      all("preprocess" in PREREQS[s] for s in STEP_ORDER if s != "preprocess"))


# --------------------------------------------------------------------------
print("\n[2] STEP_ORDER 是合法拓扑序（执行顺序的前提）")
bad = [(s, p) for s in STEP_ORDER for p in PREREQS[s]
       if STEP_ORDER.index(p) >= STEP_ORDER.index(s)]
check("每个步骤的前置都排在它前面", not bad, str(bad))
check("PREREQS 已按规范顺序排",
      all(PREREQS[s] == [x for x in STEP_ORDER if x in PREREQS[s]] for s in STEP_ORDER))


# --------------------------------------------------------------------------
print("\n[3] PREREQS 是传递闭包（置灰时要一次说清缺哪几步）")
# 只给直接依赖的话，置灰只能报"缺 RTMPose"，而 RTMPose 自己又缺检测框，
# 用户得一轮一轮试出来。这条就是判据。
check("boxes 的闭包含 preprocess", PREREQS["boxes"] == ["preprocess"], str(PREREQS["boxes"]))
check("rtmpose 的闭包含 preprocess+boxes（不是只有直接那层）",
      PREREQS["rtmpose"] == ["preprocess", "boxes"], str(PREREQS["rtmpose"]))
check("export 一路追到 preprocess",
      PREREQS["export"] == ["preprocess", "boxes", "rtmpose", "gvhmr"],
      str(PREREQS["export"]))
check("render 一路追到 preprocess（主线的 5 步全在）",
      PREREQS["render"] == MAINLINE, str(PREREQS["render"]))
# 「归并人物」和「多人轨迹」都是**可选**的一步，依赖链挂在主线之后，别的主线
# 步骤都不依赖它们。挂反了的话会出现"没开多人轨迹就没法渲染"这种事。
check("tracks 的前置就是主线那 5 步",
      PREREQS["tracks"] == MAINLINE, str(PREREQS["tracks"]))
check("merge 只要预处理 + 检测框（分组是 bbx_all.json 算的，不要主轨迹的 npz）",
      PREREQS["merge"] == ["preprocess", "boxes"], str(PREREQS["merge"]))
check("merge 在 tracks 前面（先把人并好，tracks 才不会再去跑那几段碎片）",
      STEP_ORDER.index("merge") < STEP_ORDER.index("tracks"), str(STEP_ORDER))
for s in ("tracks", "merge"):
    check(f"{s} 不在任何别步的前置里",
          all(s not in PREREQS[x] for x in STEP_ORDER), f"有步骤依赖了 {s}")
    check(f"{s} 不在默认勾选的步骤里", s not in DEFAULT_STEPS, str(DEFAULT_STEPS))
# 「归并人物」和「多人轨迹」都**不在**勾选面板里：它们的入口是「人物」面板那两个
# 按钮。这两步是同一件事的两半（把画面里的人各跑一份产物），而"谁是谁"只有人物
# 面板说得清；摆在左边会让人在两个地方来回跳，反着来还会拿到一句"没有分组决策"。
check("merge / tracks 都不在界面清单里（入口在人物面板）",
      not ({"merge", "tracks"} & set(UI_STEPS)), str(UI_STEPS))
check("界面清单只剩主线那 5 步（圆点也是 5 个）",
      UI_STEPS == MAINLINE, str(UI_STEPS))
for s in ("merge", "tracks"):
    check(f"{s} 仍然是 batch_run 的一步（作业队列按名字拼命令行）",
          s in STEP_ORDER and s in STEP_FN, f"{STEP_ORDER}")
check("人物面板那两个按钮提交的步骤都合法",
      {"merge", "tracks"} <= set(STEP_FN), "batch_run.STEP_FN 少了 merge/tracks")
check("闭包不自含", all(s not in PREREQS[s] for s in STEP_ORDER))


# --------------------------------------------------------------------------
print("\n[4] 判据是「勾了 或 盘上有」，不是「前置必须被勾」")
check("全选、盘上空的 -> 放行",
      check_prereqs(list(STEP_ORDER), products()) is None)
check("什么都不选也放行（空作业是调用方自己的事，不归依赖表管）",
      check_prereqs([], products()) is None)

r = check_prereqs(["export"], products())
check("只勾 export、盘上空的 -> 拦住", r is not None, str(r))
check("  并且点名缺的是 GVHMR 推理", r is not None and STEP_NAME["gvhmr"] in r, str(r))

# 这几个是"素材算过了，只想补后面几步"的正常用法，拦掉就会逼用户整条重跑。
# 注意判据是**整条链的产物都在**，不是只看直接上一层：
STRONG = ("preprocess", "boxes", "rtmpose", "gvhmr")
check("只勾 export、整条链的产物都在 -> 放行",
      check_prereqs(["export"], products(*STRONG)) is None)
check("只勾 gvhmr、预处理+检测框+关键点都在 -> 放行",
      check_prereqs(["gvhmr"], products("preprocess", "boxes", "rtmpose")) is None)
check("只勾 render、六步里前五步的产物都在 -> 放行",
      check_prereqs(["render"], products(*STRONG, "export")) is None)

# 链条语义的代价，明写在这里而不是留给以后的人猜：中间产物被删了，后面那步
# 会被要求把整条链补回来——哪怕照它自己的读法（export 只读 hmr4d_results.pt）
# 其实够了。选链条是因为用户要的是"前置"，而链条是最好解释的一种判据：
# 它保证跑完之后链上每个产物都在。真要跳过，命令行不受这个检查约束
# （--steps export 照跑，见 batch_run.check_prereqs 的说明）。
check("只勾 export、盘上只剩 hmr4d_results.pt -> 拦住（链条语义的代价）",
      check_prereqs(["export"], products("gvhmr")) is not None)
check("只勾 gvhmr、只有检测框和关键点、没有预处理产物 -> 拦住"
      "（GVHMR 要读那个视频，process_one 里也会挡）",
      check_prereqs(["gvhmr"], products("boxes", "rtmpose")) is not None)
check("只勾 render、只有 npz 没有预处理产物 -> 拦住（渲染要读那个视频）",
      check_prereqs(["render"], products("export")) is not None)
check("只勾 render、只有 quad 自己 -> 拦住（产物不能自证）",
      check_prereqs(["render"], products("render")) is not None)

# gvhmr 少了 rtmpose 是**软**依赖（它会退回自带的 ViTPose 跑完），但用户明确
# 要求当成硬的：能跑完却换了实现，比跑不起来更难发现。
r = check_prereqs(["gvhmr"], products("boxes"))
check("只勾 gvhmr、缺 rtmpose -> 拦住（软依赖也当硬前置）", r is not None, str(r))
check("  并且点名缺的是 RTMPose", r is not None and STEP_NAME["rtmpose"] in r, str(r))


# --------------------------------------------------------------------------
print("\n[5] 产物不能自证前置")
# 已经导出的 npz 是 export **自己的**产物，不能拿它证明 GVHMR 跑过。
# 反过来说：如果哪天有人把判据写成"有 npz 就放过 export"，这条会红。
check("有 npz 但缺 hmr4d_results.pt、只勾 export -> 拦住",
      check_prereqs(["export"], products("export")) is not None)
check("有 npz 但只勾 gvhmr -> 拦住（缺检测框/关键点）",
      check_prereqs(["gvhmr"], products("export")) is not None)
# 缺得越多，报出来越多——一次说清，别让用户改一次试一次
r = check_prereqs(["render"], products())
check("一次列出全部缺口（4 个）",
      r is not None and all(STEP_NAME[s] in r for s in PREREQS["render"]), str(r))


# --------------------------------------------------------------------------
print("\n[6] 依赖表只有一份（界面/后端拿到的必须一致）")
from server import jobs, paths  # noqa: E402

try:
    meta = jobs.steps_meta()
    check("steps_meta 带出 prereqs", meta.get("prereqs") == PREREQS)
except Exception as exc:                                   # noqa: BLE001
    check("steps_meta 带出 prereqs", False, repr(exc))

# 用一个不存在的 slug 查产物：只验键，不依赖 workspace 里实际有什么
try:
    got = paths.step_products("__no_such_slug__")
    check("step_products 的键与 STEP_ORDER 一一对应",
          list(got) == list(STEP_ORDER), f"{list(got)}")
    check("不存在的素材：所有产物都是 False", not any(got.values()), str(got))
except Exception as exc:                                   # noqa: BLE001
    check("step_products 能查", False, repr(exc))

# validate 是喂了产物的 check_prereqs，不该再有自己的规则
try:
    v = jobs.validate("__no_such_slug__", ["export"])
    check("jobs.validate 对不存在的素材只勾 export -> 拦住", v is not None, str(v))
    check("  说明与 check_prereqs 同源",
          v == check_prereqs(["export"], paths.step_products("__no_such_slug__")), str(v))
except Exception as exc:                                   # noqa: BLE001
    check("jobs.validate 能跑", False, repr(exc))


# --------------------------------------------------------------------------
print("\n[7] 读不了的预处理产物不算产物")
# 真发生过（2026-10-01 那批跑了一半）：编码写到一半被打断，MP4 的索引（moov）
# 没写出来，留下一个**大小正常但谁也解不了**的残件。判据如果只看"文件在不在"，
# 它就会被当成"预处理已完成"——第 1 步亮着、点跑被跳过，跑到第 2 步才炸，
# 报的还是一句离得很远的 `pyav can not handle the given uri`。
from server import paths as _paths                              # noqa: E402
from preprocess import video_readable                           # noqa: E402

BAD_SLUG = "ZTEST-BadProcessed"
BAD = _paths.PROCESSED / f"{BAD_SLUG}_1280p.mp4"
try:
    _paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    BAD.write_bytes(b"\x00" * 4096)          # 有大小、没有 moov
    check("残件本身读不了", not video_readable(BAD))
    check("step_products 不把它算成预处理产物",
          _paths.step_products(BAD_SLUG)["preprocess"] is False)
    check("processed_video 不会返回它", _paths.processed_video(BAD_SLUG) is None)
finally:
    BAD.unlink(missing_ok=True)

# --------------------------------------------------------------------------
print("\n[8] 原生崩溃：退出码要能读懂，而且只在这一种失败上重试")
# 真发生过：第 4 步（GVHMR 推理）开始 2 秒后整个进程被系统干掉，退出码
# 3221226505。日志里只剩一个数字，看不出是显存被占、驱动抽风，还是代码错了。
import batch_run as _br                                              # noqa: E402

check("0xC0000409 有解释", "0xC0000409" in _br.explain_exit(3221226505),
      _br.explain_exit(3221226505))
check("普通退出码原样报出", _br.explain_exit(1) == "退出码 1")
check("负数形态也认（不同版本给的有符号/无符号不一样）",
      "0xC0000409" in _br.explain_exit(-1073740791))

_real_stream, _real_log = _br.stream_proc, _br.log
_logs: list[str] = []
try:
    calls = []

    def fake_crash_then_ok(cmd, cwd, on_line):
        calls.append(1)
        return ((3221226505, ["boom"]) if len(calls) == 1 else (0, []))

    _br.stream_proc = fake_crash_then_ok
    _br.log = lambda m: _logs.append(m)
    _br.run_step(["x"], "测试步", 1, lambda f: None)
    check("原生崩溃后会重试一次并成功", len(calls) == 2, f"跑了 {len(calls)} 次")
    check("重试这件事要写在日志里",
          any("重试" in m for m in _logs), str(_logs[:2]))

    calls2 = []

    def fake_clean_fail(cmd, cwd, on_line):
        calls2.append(1)
        return (1, ["nope"])

    _br.stream_proc = fake_clean_fail
    try:
        _br.run_step(["x"], "测试步", 1, lambda f: None)
        check("退出码 1 应当抛错", False)
    except RuntimeError as exc:
        check("退出码 1 直接失败、不重试", len(calls2) == 1, f"跑了 {len(calls2)} 次")
        check("错误信息里带可读的退出码", "退出码 1" in str(exc), str(exc)[:50])
finally:
    _br.stream_proc, _br.log = _real_stream, _real_log


print(f"\n===== {len(passes)} 通过, {len(fails)} 失败 =====")
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
