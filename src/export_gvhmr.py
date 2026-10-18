"""把 GVHMR 的 hmr4d_results.pt 导出成 viz 框架能读的 npz。

为什么需要这一层
----------------
两个环境是有意分开的：
  - GVHMR 需要 Python 3.10 + torch 2.3 + pytorch3d shim
  - viz 框架需要 rerun-sdk（装在 Python 3.12）
在 3.10 环境里把结果落成中立的 npz，3.12 那边用现有的 from_smplx_npz 读进来，
两边互不污染。

导出哪些坐标系（`--spaces`，默认 `incam`）
----------------------------------------
  <out>_incam.npz   相机系重建 + K + 单位外参
                    -> 把 mesh 投影回原视频（对齐检查）+ 重投影误差诊断
                    -> **也是整条链路默认在用的那份**：算脚本、渲染主输入、
                       "这条素材导出了没有"都认它
  <out>_global.npz  世界系重建（重力对齐）
                    -> 只有 3D 视图切到世界系、渲染的世界系格子要用

**两份都是独立重建出来的**（`smpl_params_incam` / `smpl_params_global` 各跑一遍
SMPL-X 前向），不是同一份数据换坐标。每份约 0.1 MB/帧——6700 帧的素材上两份
加起来 1.5 GB，所以默认只导相机系，世界系要就显式带上。

关键点：incam 是相机坐标系（相机在原点），所以 cam_to_world = 单位阵，
直接配 K_fullimg 就能投回像素。global 是世界系，但 GVHMR 不输出对应这台相机的
真实位姿（它出图时用的是合成环绕相机），所以 global 那份不带相机、也不带 K。

用法（在 gvhmr 环境里跑）
    python export_gvhmr.py GVHMR/outputs/demo/dance_1280p/hmr4d_results.pt
    python export_gvhmr.py <pt> --spaces incam,global     # 两个都要
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/
ROOT = HERE.parent                              # 项目根
GVHMR_ROOT = ROOT / "GVHMR"                     # 上游仓库留在根目录
SHIMS = HERE / "shims"                          # shim 随代码放在 src/ 下

# 网格文件的命名规则（`x_incam.npz` -> `x_incam.mesh.npz`）。渲染端和批处理也
# import 同一份——规则抄第二遍就会漂。
sys.path.insert(0, str(HERE))
from mesh_npz import mesh_path_for  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="导出 GVHMR 结果供 viz 框架使用")
    ap.add_argument("results", help="hmr4d_results.pt 路径")
    ap.add_argument("--preprocess-dir", default=None,
                    help="preprocess 目录（默认从 results 路径推断）")
    ap.add_argument("--out", default=None, help="输出前缀（默认与 results 同目录）")
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="ViTPose 置信度低于此值视为遮挡（默认 0.5）")
    ap.add_argument("--fps", type=float, default=30.0)
    # 导出哪些坐标系。**默认只导相机系**：它是整条链路都在用的那一份
    # （算脚本的默认来源、渲染的主输入、results_dir 的判据都认它），而且带相机
    # 内外参、诊断全可用。世界系是**另一套独立重建**（不是前者的刚性变换），
    # 只有 3D 视图切到世界系、渲染的世界系格子要用。两份各约 0.1 MB/帧——
    # 6700 帧的素材上两份加起来 1.5 GB，只导相机系就省掉一半。
    ap.add_argument("--spaces", default="incam",
                    help="导出哪些坐标系：incam（默认）/ global / 逗号分隔两个都写")
    args = ap.parse_args()

    spaces = {s.strip() for s in args.spaces.split(",") if s.strip()}
    if not spaces or spaces - {"incam", "global"}:
        print(f"--spaces 只能是 incam / global（给的是 {args.spaces!r}）", file=sys.stderr)
        return 2

    results_path = Path(args.results).resolve()
    if not results_path.exists():
        print(f"找不到结果文件：{results_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(SHIMS))
    sys.path.insert(0, str(GVHMR_ROOT))
    os.chdir(GVHMR_ROOT)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    from hmr4d.utils.smplx_utils import make_smplx

    # ---------- 读结果 ----------
    pred = torch.load(results_path, map_location="cpu")
    pre_dir = Path(args.preprocess_dir) if args.preprocess_dir else results_path.parent / "preprocess"

    K = pred["K_fullimg"].float()                         # (L, 3, 3)
    L = K.shape[0]
    print(f"帧数: {L}")

    # ---------- SMPL-X 前向 ----------
    # 上游 render 函数用的就是这套组合，保持一致
    model = make_smplx("supermotion").cpu().eval()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", map_location="cpu")
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", map_location="cpu")
    faces_smplx = np.asarray(model.faces, dtype=np.uint32)
    print(f"SMPL-X: {len(faces_smplx)} 面, J_regressor {tuple(J_regressor.shape)}")

    def reconstruct(params: dict) -> tuple[np.ndarray, np.ndarray]:
        """SMPL-X 参数 -> (SMPL-X 顶点 (L,V,3), SMPL 关节 (L,J,3))。

        注意 BodyModelSMPLX 期望**扁平的 2D 输入** (B, dim)，帧数就是 batch 维——
        不要传 (1, L, dim)。它的包装层会以 max(len(var)) 推断 batch，
        再展开 jaw/hand 等默认姿态；传 3D 会让 batch 被算成 1，与 L 对不上而报错。
        """
        with torch.no_grad():
            out = model(
                body_pose=params["body_pose"].float(),          # (L, 63)
                betas=params["betas"].float(),                  # (L, 10)
                global_orient=params["global_orient"].float(),  # (L, 3)
                transl=params["transl"].float(),                # (L, 3)
            )
            verts_x = out.vertices                              # (L, 10475, 3)
            # smplx2smpl 是稀疏矩阵，einsum 不支持。sparse.mm 要求 dense 侧第一维
            # 等于矩阵的列数(10475)，所以把顶点维提到前面，一次性算完所有帧。
            L, V = verts_x.shape[0], verts_x.shape[1]
            flat = verts_x.permute(1, 0, 2).reshape(V, -1)       # (10475, L*3)
            verts_smpl = torch.sparse.mm(smplx2smpl, flat)       # (6890, L*3)
            verts_smpl = verts_smpl.reshape(-1, L, 3).permute(1, 0, 2)  # (L, 6890, 3)
            joints = torch.einsum("j v, l v c -> l j c", J_regressor, verts_smpl)
        return verts_x.numpy().astype(np.float32), joints.numpy().astype(np.float32)

    # 只重建要写的那几份：reconstruct 是 SMPL-X 逐帧前向，是整个导出里最贵的
    # 一步，白算一份世界系等于多花一半时间、多写 0.75 GB。
    verts_incam = joints_incam = verts_glob = joints_glob = None
    if "incam" in spaces:
        print("重建 incam ...")
        verts_incam, joints_incam = reconstruct(pred["smpl_params_incam"])
    if "global" in spaces:
        print("重建 global ...")
        verts_glob, joints_glob = reconstruct(pred["smpl_params_global"])
    skipped = {"incam", "global"} - spaces
    if skipped:
        print(f"（这次不导：{'、'.join(sorted(skipped))}。要补上就再跑一次 "
              f"--spaces {','.join(sorted(spaces | skipped))}）")

    # ---------- 2D 证据 ----------
    kp2d = conf = None
    vitpose_path = pre_dir / "vitpose.pt"
    if vitpose_path.exists():
        vp = torch.load(vitpose_path, map_location="cpu").float().numpy()   # (L, 17, 3)
        kp2d, conf = vp[:, :, :2], vp[:, :, 2]
        visible = conf >= args.conf_threshold
        print(f"ViTPose: {kp2d.shape[1]} 个关键点，"
              f"平均置信度 {conf.mean():.3f}，低置信（视为遮挡）占比 {(~visible).mean():.1%}")
    else:
        print(f"警告：找不到 {vitpose_path}，跳过 2D 关键点")

    # COCO17 -> SMPL24 关节索引。**必须显式给出**：两套顺序完全不同
    # （COCO 第 0 是鼻子，SMPL 第 0 是骨盆），靠索引对齐会让重投影误差大一个数量级。
    #
    # -1 = 无对应，重投影诊断会跳过：
    #   眼、耳：SMPL 没有对应关节
    #   鼻子：虽然常被映射到 SMPL 的 head 关节，但 head 关节在颅底、鼻子在面部前方，
    #         两者存在固定的解剖偏移，会稳定地制造 ~35px 的假信号，淹没真问题。
    #
    # SMPL24: 0骨盆 1左胯 2右胯 3spine1 4左膝 5右膝 6spine2 7左踝 8右踝
    #         9spine3 10左足 11右足 12颈 13左锁 14右锁 15头 16左肩 17右肩
    #         18左肘 19右肘 20左腕 21右腕 22左手 23右手
    COCO17_TO_SMPL24 = np.array(
        [-1, -1, -1, -1, -1, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8],
        dtype=np.int32,
    )

    # ---------- 输出 ----------
    out_prefix = Path(args.out).resolve() if args.out else results_path.parent / results_path.stem

    # 画面尺寸从内参反推（cx,cy 是主点，约等于 宽/高 的一半）。
    # 不要硬编码——竖屏素材很容易写成横向。
    img_w = int(round(2 * float(K[0, 0, 2])))
    img_h = int(round(2 * float(K[0, 1, 2])))
    print(f"画面尺寸（由 K 反推）: {img_w} x {img_h}")

    def save(path: Path, **fields):
        payload = {k: v for k, v in fields.items() if v is not None}
        payload["fps"] = np.array([args.fps], dtype=np.float32)
        payload["image_size"] = np.array([img_w, img_h], dtype=np.int32)
        np.savez_compressed(path, **payload)
        print(f"  写入 {path}  ({path.stat().st_size/1024/1024:.1f} MB)")

    def save_mesh(path: Path, verts, **fields):
        """网格单独一个文件——**它一个就占了原 npz 的 99.5%**（10475 顶点 × 3
        × float32 = 每帧 123 KB），而只有渲染和 viz 读它。3D 视图、算脚本、
        `motion` 接口都只用 joints（每帧 300 字节）与 2D 关键点。捆在一起的话，
        一条不渲染的素材也要一直占着几百 MB~1 GB。

        拆开之后"不渲染就不占"才成为可能：删掉这些 .mesh.npz 即可，
        要看的时候用同一份 .pt 重跑一次本脚本（20~60 秒）就长回来。
        """
        payload = {"vertices": verts, "faces": faces_smplx}
        payload.update({k: v for k, v in fields.items() if v is not None})
        np.savez_compressed(path, **payload)
        print(f"  写入 {path}  ({path.stat().st_size/1024/1024:.1f} MB)  ← 网格"
              f"（只有渲染/viz 读它，不需要就删）")

    incam_path = out_prefix.with_name(out_prefix.name + "_incam.npz")
    glob_path = out_prefix.with_name(out_prefix.name + "_global.npz")

    print("导出 ...")
    if verts_incam is not None:
        save(
            incam_path,
            joints=joints_incam,
            K=K.numpy().astype(np.float32),
            cam_to_world=np.tile(np.eye(4, dtype=np.float32), (L, 1, 1)),  # incam: 相机在原点
            keypoints_2d=kp2d,
            keypoints_conf=conf,
            kp_to_joint=COCO17_TO_SMPL24,
        )
        save_mesh(mesh_path_for(incam_path), verts_incam)
    if verts_glob is not None:
        save(
            glob_path,
            joints=joints_glob,
            keypoints_2d=kp2d,
            keypoints_conf=conf,
            kp_to_joint=COCO17_TO_SMPL24,
        )
        save_mesh(mesh_path_for(glob_path), verts_glob)

    print()
    print("完成。用 viz 框架看：")
    if verts_incam is not None:
        print(f"  python run_viz.py --npz \"{incam_path}\"        # 相机系，可叠加回原视频")
    if verts_glob is not None:
        print(f"  python run_viz.py --npz \"{glob_path}\"         # 世界系，看足部/轨迹")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
