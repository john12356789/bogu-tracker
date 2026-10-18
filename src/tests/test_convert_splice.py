"""「单独重算只替换这一段」的验证。

    python src/tests/test_convert_splice.py

盯住四件事：
  1. **段外一个字节都不能动**——这是这个功能的全部意义。做法：整条先算一次，
     再用**改了参数**的同一段算一次，然后逐 action 比对段外部分。
  2. tcode 的行号就是绝对帧号（一帧一行），所以"替换哪几行"= "替换哪几帧"，
     行数必须和 funscript 的 action 数一致。
  3. **算不出来时预览目录要原封不动**：原来是先 _wipe 再算，算挂了只剩半截，
     而 /promote 是把整个预览目录提交成正式产物的——等于把其余段删了。
  4. 目录里那份本身不是整条（旧版留下的半截）时，要**拒绝**替换并让用户先
     重算，而不是拿半截当基准、把缺的部分永久留在产物里。

全程跑在临时目录上（preview_dir 被打桩），不碰用户的草稿。
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from server import convert, paths  # noqa: E402

SLUG = "BV1RZY4zPEw2"          # 双人素材，有 __t 轨迹目录，是这条功能的主场景
SPACE = "incam"
FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def _actions(d: Path, axis: str = "L0") -> list[dict]:
    f = paths.funscript_files_in(d).get(axis)
    if f is None:
        return []
    return json.loads(f.read_text(encoding="utf-8"))["actions"]


def _names(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir() if p.is_file())


def _prefixes(d: Path) -> set[str]:
    """脚本文件的名字前缀（不含 person_timeline.json 这类非脚本文件）。"""
    return {p.name.split(".")[0] for p in d.iterdir()
            if p.is_file() and (p.name.endswith(".funscript")
                                or p.name.endswith(".tcode.txt"))}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="splice_test_"))
    real = convert.preview_dir
    convert.preview_dir = lambda slug: tmp          # 打桩：不碰真草稿
    try:
        return _run(tmp)
    finally:
        convert.preview_dir = real
        shutil.rmtree(tmp, ignore_errors=True)


def _run(tmp: Path) -> int:
    print(f"[临时预览目录] {tmp}")

    # ---- 1. 整条 ----
    print("\n1) 整条重算（主轨迹）")
    r = convert.convert(SLUG, {"space": SPACE, "person_tid": 2})
    if not r.get("ok"):
        print("  !! 整条都算不出来，后面的检查没有意义：", r.get("error"))
        return 1
    base = _actions(tmp)
    tcode_base = convert._tcode_lines(tmp)
    n = len(base)
    print(f"  {n} 帧，at {base[0]['at']}~{base[-1]['at']}，tcode {len(tcode_base)} 行")
    check(n > 1000, f"整条脚本是全长的（{n} 帧）")
    check(base[0]["at"] == 0, "整条从第 0 帧起")
    check(len(tcode_base) == n, "tcode 行数 == action 数（一帧一行）")
    name0 = _names(tmp)[0]
    print(f"  产物名 {name0}")

    # ---- 2. 只替换一段（参数故意改掉，好分辨哪部分是新算的）----
    #
    # 取**中间**一段：两侧都有段外的内容，"只替换这一段"才算真的验到了。
    # 边界直接用**下标**比，不用时间——一帧一个 action，下标就是帧号，
    # 而按毫秒比会碰到 at=round(i*1000/fps) 的取整边界（22033 vs 22033.33），
    # 那比的是取整误差，不是这个功能。
    print("\n2) 只替换第 661~1500 帧（高通 0.5 -> 1.0）")
    a, b = 661, 1500
    r2 = convert.convert(SLUG, {"space": SPACE, "person_tid": 2,
                                "highpass": 1.0, "range": [a, b]})
    check(r2.get("ok"), f"替换成功（{r2.get('error') or 'ok'}）")
    now = _actions(tmp)
    tcode_now = convert._tcode_lines(tmp)
    check(not r2.get("error"), "没有 error")
    check(len(now) == n, f"整条还在：{len(now)} 帧 == {n}（不是只剩 {b - a} 帧）")
    check(len(tcode_now) == n, f"tcode 行数 {len(tcode_now)} == {n}")
    check(now[:a] == base[:a], f"替换段之前 {a} 个 action 逐个相同")
    check(tcode_now[:a] == tcode_base[:a], f"tcode 前 {a} 行原样")
    check(now[b:] == base[b:], f"替换段之后 {n - b} 个 action 逐个相同")
    check(tcode_now[b:] == tcode_base[b:], f"tcode 后 {n - b} 行原样")

    mid_new, mid_old = now[a:b], base[a:b]
    diff = sum(1 for p, q in zip(mid_new, mid_old) if p["pos"] != q["pos"])
    print(f"  中间 {len(mid_new)} 个 action 里有 {diff} 个值和旧的不同")
    check(len(mid_new) == b - a, f"替换段长度 {len(mid_new)} == {b - a}")
    check(diff > 0, "被替换的那一段确实用了新参数算出来的值")
    check(all(p["at"] == q["at"] for p, q in zip(mid_new, mid_old)),
          "时间戳一格没挪（只换值，不动时间轴）")

    # multi 合体里每条轴都在
    multi = next(p for p in tmp.iterdir() if p.name.endswith(".multi.funscript"))
    ax = {x["id"]: x["actions"] for x in json.loads(multi.read_text(encoding="utf-8"))["axes"]}
    check(set(ax) == {"L0", "L1", "L2", "R0", "R1", "R2"}, f"multi 里六条轴都在：{sorted(ax)}")
    check(len(ax["R2"]) == n, "multi 里的 R2 也是全长")

    # 目录里只有一套名字（换过轨迹会换前缀，旧的必须清掉）
    pref = _prefixes(tmp)
    check(pref == {name0.split(".")[0]}, f"目录里只有一套名字：{pref}")

    # ---- 3. 半截脚本要拒绝，而不是当基准 ----
    #
    # 旧版"单独重算"留下的就是半截（目录里那份开头不是第 0 帧）。拿它当基准
    # 替换的话，缺的那部分会**永久留在产物里**，而且看起来一切正常。
    # 半截有两种形态，都要挡住。第二种（从 0 开始但提前结束）正是老版
    # 「单独重算」留下的样子，只查开头是不是第 0 帧是挡不住的。
    seg = Path(tempfile.mkdtemp(prefix="seg_fake_"))
    convert._write_axes(seg, "seg", {"L0": base[1500:1600]})
    try:
        for what, half in (("开头不是第 0 帧", base[661:1500]),
                           ("从 0 开始但提前结束", base[:661])):
            print(f"\n3) 目录里那份本身只是半截（{what}）")
            shutil.rmtree(tmp)
            tmp.mkdir()
            convert._write_axes(tmp, "half", {"L0": half})
            snap = _names(tmp)
            err = convert._splice_segment(tmp, seg, len(base))
            check(err is not None and "整条" in err, f"拒绝替换并给出说明：{err}")
            check(_names(tmp) == snap, "拒绝时目录一个文件都没动")
            check(_actions(tmp, "L0") == half, "拒绝时内容一个字节没改")
    finally:
        shutil.rmtree(seg, ignore_errors=True)

    # ---- 4. 段算失败时预览原封不动 ----
    print("\n4) 段算不出来时预览不动")
    shutil.rmtree(tmp)
    tmp.mkdir()
    convert._write_axes(tmp, "full", {"L0": base})
    snap = _names(tmp)
    before = _actions(tmp, "L0")
    r4 = convert.convert(SLUG, {"space": SPACE, "person_tid": 2,
                                "range": [10, 20], "highpass": "不是数字"})
    check(not r4.get("ok"), "返回 ok=False")
    check(bool(r4.get("error")), f"带 error 说明：{r4.get('error')}")
    check(_names(tmp) == snap, "预览目录文件没变")
    check(_actions(tmp, "L0") == before, "预览内容一个字节没改")

    # ---- 5. 时段指定整条（_merge_segments 被这次改动重构过，得跑一遍）----
    print("\n5) 时段指定整条（0~661 用轨迹 2，之后用轨迹 8）")
    shutil.rmtree(tmp)
    tmp.mkdir()
    r5 = convert.convert(SLUG, {"space": SPACE,
                                "person_segs": [{"from": 0, "tid": 2},
                                                {"from": 661, "tid": 8}]})
    check(r5.get("ok"), f"整条算出来了（{r5.get('error') or 'ok'}）")
    arts = _actions(tmp)
    check(len(arts) == n, f"拼出来还是全长 {len(arts)} == {n}")
    check(_actions(tmp, "R2") and len(_actions(tmp, "R2")) == n, "R2 也是全长")
    check(len(convert._tcode_lines(tmp)) == n, f"tcode {len(convert._tcode_lines(tmp))} 行")
    # 产物名字要取自 npz 所在目录名（和 to_funscript 一致），不是 npz.stem——
    # 否则同一次「重算」会因为时段开关开没开而写出两个不同名字的同一份脚本。
    got = _prefixes(tmp)
    check(got == {name0.split(".")[0]},
          f"名字和整条重算一致：{got} == {{{name0.split('.')[0]}}}")

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
