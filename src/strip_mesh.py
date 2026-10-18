"""把已有结果 npz 里的网格（vertices/faces）摘掉，只留关节/关键点/相机。

    python src/strip_mesh.py            # 只看，不动（默认）
    python src/strip_mesh.py --apply    # 真做

**为什么**：vertices 一个就占结果文件的 **99.5%**（10475 顶点 × 3 × float32 =
每帧 123 KB），而只有渲染和 viz 读它——3D 视图、算脚本、前端的 motion 接口都只用
joints（每帧 300 字节）和 2D 关键点。实测盘上 58 份 npz 共 33.7 GB，其中 33.5 GB
是这份网格，而 52 条素材里只有 8 条渲染过。拆开（见 mesh_npz.py）之后
"不渲染就不占"才成立：要渲染时 batch_run 会自动用同一份 .pt 补回来（20~60 秒）。

**逐字节不丢东西的写法**：不用 np.load 把整个 npz 读出来再另存——那会把
846 MB 的 vertices 也解压一遍（58 份就是十几分钟）。这里只遍历小的那几个成员，
顶点根本不碰（不 inflate）。

三道保险：
  1. **同目录必须有 hmr4d_results.pt**。网格由它重建，它不在就永远补不回来，
     那种一条都不动。
  2. 已经有 .mesh.npz 的（新格式导出的）跳过。
  3. 写完之后**读回来验**：键齐、形状对，才算成功，然后才原子替换。

删掉的只是"能重建出来的东西"，不是原始数据。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from mesh_npz import mesh_path_for  # noqa: E402
from server import paths  # noqa: E402

MESH_KEYS = ("vertices", "faces")


def main() -> int:
    ap = argparse.ArgumentParser(description="摘掉结果 npz 里的网格")
    ap.add_argument("--apply", action="store_true", help="真的改（不加就只报告）")
    ap.add_argument("--only", default=None, metavar="SLUG",
                    help="只处理这条素材（先拿一条验证全链路，再批量）")
    args = ap.parse_args()

    def is_result(p: Path) -> bool:
        """只认真正的结果文件：排除拆出去的网格、以及本脚本自己的临时文件。

        **这个过滤是必须的**：glob 的 `*.npz` 会把 `x.npz.mesh.npz` 和
        `x.npz.tmp.npz` 一起捞进来，然后脚本会去"处理"它们——踩过一次，
        结果是它去摘自己刚写的临时文件。
        """
        n = p.name
        return ".mesh.npz" not in n and ".tmp.npz" not in n and ".strip-tmp" not in n

    npzs = sorted(p for p in paths.OUTPUTS.glob("*/**/hmr4d_results_*.npz") if is_result(p))
    if args.only:
        npzs = [p for p in npzs if p.relative_to(paths.OUTPUTS).parts[0] == args.only]
    if not npzs:
        print("没找到结果 npz")
        return 0

    free0 = shutil.disk_usage(str(paths.OUTPUTS)).free
    done: list[tuple[Path, int, int]] = []
    skipped: list[tuple[Path, str]] = []

    for p in npzs:
        if mesh_path_for(p).exists():
            skipped.append((p, "已经是新格式（有 .mesh.npz）"))
            continue
        if not (p.parent / "hmr4d_results.pt").exists():
            skipped.append((p, "同目录没有 hmr4d_results.pt——网格补不回来，不动"))
            continue
        try:
            with np.load(p) as z:
                keys = [k for k in z.files if k not in MESH_KEYS]
                if len(keys) == len(z.files):
                    skipped.append((p, "本来就没有网格"))
                    continue
                payload = {k: z[k] for k in keys}      # 只解压小成员
                old_size = p.stat().st_size
        except (OSError, ValueError) as exc:
            skipped.append((p, f"读不了：{exc}"))
            continue

        if not args.apply:
            # 只估算：新文件大小的下界 = 各小成员的解压前字节数
            est = sum(v.nbytes for v in payload.values())
            done.append((p, old_size, max(est, 1)))
            continue

        # 临时名**必须以 .npz 结尾**：np.savez_compressed 见路径不以 .npz 结尾
        # 会自己补一个，于是 `x.npz.strip-tmp` 写出的是 `x.npz.strip-tmp.npz`，
        # replace 的目标却还是前者 —— 报"文件不存在"，真正写出的那个成了孤儿。
        tmp = p.with_name(p.name + ".tmp.npz")
        try:
            np.savez_compressed(tmp, **payload)
            with np.load(tmp) as z2:                   # 验：键齐、形状对
                for k, v in payload.items():
                    if k not in z2.files or z2[k].shape != v.shape:
                        raise ValueError(f"回读校验失败：{k}")
            new_size = tmp.stat().st_size
            tmp.replace(p)                             # 原子替换
        except (OSError, ValueError) as exc:
            for q in (tmp, tmp.with_name(tmp.name + ".npz")):
                q.unlink(missing_ok=True)              # 两种可能的落点都清掉
            skipped.append((p, f"写入失败，原文件保留：{exc}"))
            continue
        done.append((p, old_size, new_size))

    for p, old, new in done:
        print(f"  {'已摘' if args.apply else '可省'} {(old-new)/1e6:8.1f} MB  "
              f"({old/1e6:.0f} -> {new/1e6:.0f} MB)  {p.relative_to(paths.OUTPUTS)}")
    for p, why in skipped:
        print(f"  跳过  {p.relative_to(paths.OUTPUTS)}  —— {why}")

    saved = sum(o - n for _, o, n in done)
    print()
    verb = "已释放" if args.apply else "可释放（加 --apply 执行）"
    print(f"{len(done)} 份，{verb} {saved/1e9:.2f} GB；{len(skipped)} 份跳过")
    if args.apply:
        free1 = shutil.disk_usage(str(paths.OUTPUTS)).free
        print(f"可用空间 {free0/1e9:.2f} -> {free1/1e9:.2f} GB")
    print("\n网格要补回来：渲染时会自动补（batch_run 的 _ensure_mesh），"
          "或手敲 python src/export_gvhmr.py <hmr4d_results.pt> --spaces incam")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
