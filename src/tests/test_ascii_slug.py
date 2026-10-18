"""素材短名（slug）必须是**每一条素材各自的**身份。

    python src/tests/test_ascii_slug.py

背景（这就是要防的那个 bug）
---------------------------
`ascii_slug` 原来的最后一行是 `return slug or "video"`。取名字开头的 ASCII 段
是为了喂给 hydra（它不接受非 ASCII），这条没问题；但兜底值 **"video" 不是任何
一条素材的身份**——凡是名字开头不是 ASCII 的素材（中文标题）全都映射到同一个
"video"，于是：

    outputs/video/                     ← 所有中文标题素材的重建产物挤在一个目录
    processed/video_1280p_30fps.mp4    ← 后跑的把先跑的整个顶掉

实测 workspace/videos/ 下 52 条素材里 `旗袍—卡点爆摇.mp4` 就落在这里（其余
全部是 BV 号开头）。用户看到的现象是"只有英文开头的能跑对，其余的输出成一个
叫 video 的文件互相覆盖"——不是编码问题，是这个正则只匹配**开头**的 ASCII 段。

这里盯住三件事：

  1. **已有素材的 slug 一个都不许变**。slug 是 52 条素材在盘上的身份（目录名、
     npz 名、脚本名全由它拼出来），规则一动它们就集体"失联"。
  2. 兜底必须唯一且稳定：同一个名字永远同一个 slug，不同的名字必定不同。
  3. **反查不能用前缀匹配**。slug 只在"名字开头是 ASCII"时才是文件名的前缀；
     而且 `BV037*` 会连 `BV0370-xxx.mp4` 一起匹配上——删除路径上那就是删错文件。
"""
import sys
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# 输出走 UTF-8：管道里 Python 会退回 GBK，中文和 ✓/✗ 直接 UnicodeEncodeError
# （在报错那行崩掉，看到的是一堆 traceback 而不是"哪几项没过"）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from batch_run import ascii_slug            # noqa: E402
from server import cleanup, paths           # noqa: E402

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


# 真实存在于 workspace/videos/ 的名字。**这些是回归基线**：它们的 slug 已经在
# 盘上（outputs/<slug>/、processed/<slug>_1280p_30fps.mp4），改了就对不上。
REAL_NAMES = [
    ("BV037-兔娘-十二月月票专属舞蹈 No No No", "BV037"),
    ("BV09234-(VAM)Ariooo-1", "BV09234"),
    ("BV0985-欣小萌", "BV0985"),
    ("BV12xZw8kEYE-每天都想吃小鱼干-不知道 我的身材很曼妙", "BV12xZw8kEYE"),
    # 注意尾巴上的 "-Lo"：连字符也在 ASCII 段里，所以开头的 ASCII 段一直吃到
    # "…-Lo"。盘上的目录就是 outputs/BV1KJes6KEhJ-Lo/ —— 基线要以**实际产物**
    # 为准，不能凭"BV 号到哪结束"想当然。
    ("BV1KJes6KEhJ-Lo老师-s信版-解锁新场景！枣子姐的2分钟瑜伽训练vlog", "BV1KJes6KEhJ-Lo"),
]


def _run() -> int:
    print("\n[1] 回归：已有素材的 slug 不许变")
    for name, want in REAL_NAMES:
        got = ascii_slug(name)
        check(got == want, f"{name[:24]}… -> {got!r}（应为 {want!r}）")

    print("\n[2] 中文标题：不能再落到 'video'")
    zh = "旗袍—卡点爆摇"
    got = ascii_slug(zh)
    check(got != "video", f"{zh} -> {got!r}，不是 'video'")
    check(len(got) > 0, "非空")
    check(bool(cleanup.SAFE_SLUG.match(got)),
          f"{got!r} 匹配 SAFE_SLUG（slug 来自 URL，要能过清理路径的校验）")

    print("\n[3] 兜底必须稳定且唯一")
    check(ascii_slug(zh) == ascii_slug(zh), "同一个名字两次结果相同（产物才找得回来）")
    other = ascii_slug("旗袍—卡点爆摇 第二支")
    check(other != got, f"不同名字不同 slug：{other!r} != {got!r}")
    # 原实现下这两个和"任何中文名"全是同一个 'video'，这正是互相覆盖的原因
    check(ascii_slug("新素材") != ascii_slug("旧素材"), "两个中文名不共用 slug")
    check(ascii_slug("") != "video", "空名也不许落到 'video'")

    print("\n[4] 太短的 ASCII 段不构成身份")
    # "1分40秒-…" 开头只有 1 个 ASCII 字符；另一个 "1月新番-…" 会撞上同一个 '1'
    check(ascii_slug("1分40秒-BV037") != ascii_slug("1月新番-BV037"),
          f"两个数字开头的名字不共用 slug：{ascii_slug('1分40秒-BV037')!r}")

    print("\n[5] 反查 slug -> 原片：按规则算，不做前缀匹配")
    fn = getattr(paths, "videos_for", None)
    if fn is None:
        check(False, "paths.videos_for 还不存在（反查仍是前缀 glob）")
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="slugtest_"))
    try:
        for n in ("BV037-兔娘.mp4", "BV0370-另一个素材.mp4", "旗袍—卡点爆摇.mp4"):
            (tmp / n).write_bytes(b"x")
        got = [p.name for p in fn(ascii_slug("旗袍—卡点爆摇"), root=tmp)]
        check(got == ["旗袍—卡点爆摇.mp4"], f"中文名反查命中：{got}")
        got = [p.name for p in fn("BV037", root=tmp)]
        check(got == ["BV037-兔娘.mp4"],
              f"BV037 只命中自己，**不**连 BV0370 一起返回：{got}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def main() -> int:
    _run()
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
