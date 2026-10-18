"""「归并人物」这一步的编排（不跑 GPU：把 run_step 换成记账的）。

    python src/tests/test_merge_step.py

这一步本身不重建，它是**把几段的检测框并成一条，然后当一条轨迹重跑 2~5 步**。
所以要盯的都不是算法，而是编排：

  1. **没有分组决策时必须什么都不做**——不管是"从没决定过"（没有文件）还是
     "撤销了"（空文件）。归并是**要人点头**的操作（见 tracklet_merge.effective），
     算法建议只是界面上的东西，不能自己动手跑一遍全片重建。
  2. 有决策时命令行长什么样：`--merge-tids`、用最小 tid 当输出根、复用
     bbx_all.json（不重跑 YOLO）。
  3. **分组变了必须作废旧产物**。每一步都是"文件在就跳过"，不作废的话跑完
     还是按旧框算出来的东西，界面上却显示"跑完了"——这是最难查的一类错。
  4. 决策里出现了这条素材没有的 tid（换过预处理产物 / 删过轨迹）时跳过那一组，
     而不是整条失败。
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import batch_run as br            # noqa: E402
from tracklet_merge import save   # noqa: E402

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def make_ctx(tmp: Path) -> dict:
    """造一条素材的目录形状：outputs/<slug>/gvhmr_rtmpose/<stem>/bbx_all.json"""
    slug, stem = "ZTEST-Merge", "ZTEST-Merge_1280p_30fps"
    out_root = tmp / "outputs" / slug
    rt_root = out_root / "gvhmr_rtmpose"
    rt_dir = rt_root / stem
    (rt_dir / "preprocess").mkdir(parents=True, exist_ok=True)
    segs = [{"tid": 1, "frames": list(range(0, 50)), "coverage": 0.5,
             "boxes": [[10.0, 10.0, 50.0, 110.0]] * 50, "runs": [[0, 49]],
             "area_sum": 1.0},
            {"tid": 8, "frames": list(range(60, 110)), "coverage": 0.5,
             "boxes": [[12.0, 10.0, 52.0, 110.0]] * 50, "runs": [[60, 109]],
             "area_sum": 1.0},
            # 多一条，好让"分组变了"有真东西可变（1,8 -> 1,2,8）
            {"tid": 2, "frames": list(range(52, 59)), "coverage": 0.06,
             "boxes": [[11.0, 10.0, 51.0, 110.0]] * 7, "runs": [[52, 58]],
             "area_sum": 0.1}]
    (rt_dir / "bbx_all.json").write_text(json.dumps(
        {"version": 1, "n_frames": 120, "width": 100, "height": 120,
         "segments": segs, "dropped": []}), encoding="utf-8")
    return {"out_root": out_root, "rt_root": rt_root, "rt_dir": rt_dir,
            "proc_video": tmp / "processed" / f"{stem}.mp4", "step_opts": {},
            "force": False}


def _run() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mergestep_"))
    calls: list[list[str]] = []
    real_run_step = br.run_step
    br.run_step = lambda cmd, label, idx, on_prog: calls.append([str(c) for c in cmd])
    try:
        ctx = make_ctx(tmp)
        f = ctx["out_root"] / "people_merge.json"

        print("\n[1] 撤销归并（空决策文件）-> 空步（一个子进程都不许起）")
        save(f, [])                      # 空文件 = "别并"，不是"还没决定"
        calls.clear()
        br._run_merged_groups(ctx, lambda f: None)
        check(calls == [], f"没起任何子进程（起了 {len(calls)} 个）")

        print("\n[1b] 没有决策文件 -> 同样是空步（**归并默认关闭**）")
        f.unlink()
        calls.clear()
        br._run_merged_groups(ctx, lambda f: None)
        check(calls == [], f"没起任何子进程（起了 {len(calls)} 个）")

        print("\n[2] 有决策 -> 一组跑 4 步，且用最小 tid 当输出根")
        save(f, [{"tids": [1, 8]}])
        calls.clear()
        br._run_merged_groups(ctx, lambda f: None)
        check(len(calls) == 4, f"跑了 4 步（实际 {len(calls)}）")
        prep = next((c for c in calls if "prep_boxes.py" in c[1]), [])
        check("--merge-tids" in prep and prep[prep.index("--merge-tids") + 1] == "1,8",
              f"prep_boxes 带 --merge-tids 1,8：{prep[-6:]}")
        check("--tracks-from" in prep, "复用 bbx_all.json（不重跑 YOLO）")
        check("gvhmr_rtmpose__t1" in " ".join(prep), f"输出根用最小 tid：{prep[3]}")
        exp = [c for c in calls if "export_gvhmr.py" in " ".join(c)]
        check(bool(exp) and "gvhmr_rtmpose__t1" in " ".join(exp[0]), "导出也落在同一个根")
        marker = ctx["out_root"] / "gvhmr_rtmpose__t1" / "merge_spec.json"
        check(marker.exists() and json.loads(marker.read_text()) == [1, 8],
              f"这一根的分组凭证写下了：{marker.exists()}")

        print("\n[3] 分组变了 -> 作废旧产物（否则跑完还是旧框算出来的东西）")
        stale = [ctx["out_root"] / "gvhmr_rtmpose__t1" / ctx["proc_video"].stem / "preprocess" / n
                 for n in ("bbx.pt", "vit_features.pt", "vitpose.pt")]
        for p in stale:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stale")
        old_npz = stale[0].parent.parent / "hmr4d_results_incam.npz"
        old_npz.write_bytes(b"stale")
        save(f, [{"tids": [1, 8]}])            # 先确认"一样的规格"不清
        calls.clear()
        br._run_merged_groups(ctx, lambda f: None)
        check(all(p.exists() for p in stale), "分组没变 -> 旧产物**不许**动（否则每次白跑）")
        save(f, [{"tids": [1, 2, 8]}])
        calls.clear()
        br._run_merged_groups(ctx, lambda f: None)
        check(not any(p.exists() for p in stale),
              f"分组变了 -> bbx/特征/关键点被清掉（还剩 {[p.name for p in stale if p.exists()]}）")
        check(not old_npz.exists(), "旧的 npz 也清掉（不然导出会跳过，留下的还是旧动作）")

        print("\n[4] 决策里有这条素材没有的 tid -> 跳过那一组，不许整条失败")
        save(f, [{"tids": [1, 8]}, {"tids": [31, 44]}])
        calls.clear()
        br._run_merged_groups(ctx, lambda f: None)
        check(len(calls) == 4, f"只跑了存在的那一组（实际起了 {len(calls)} 个子进程）")
        check(all("gvhmr_rtmpose__t1" in " ".join(c) or "prep_boxes.py" in " ".join(c)
                  for c in calls), "全都落在 t1 那个根里，没有去碰 31/44")
    finally:
        br.run_step = real_run_step
        shutil.rmtree(tmp, ignore_errors=True)


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
