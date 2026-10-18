"""在原生 Windows 上跑 GVHMR 推理，跳过 pytorch3d 渲染。

为什么需要这个入口
------------------
1. Meta 从没发布过 pytorch3d 的 Windows wheel（已实测：Linux 版 200，Windows 版 403），
   而 GVHMR 的 requirements.txt 直接锁死了那个 linux_x86_64 直链。
2. 但 pytorch3d 的 CUDA 渲染器**只用于出图**——推理结果在渲染之前就已经存盘了。
   推理路径真正需要的只有 pytorch3d.transforms（纯旋转数学）和 ops.knn。

所以：用 shims/ 里的纯 PyTorch 实现顶上那两个模块，然后只跑到预测为止。
可视化交给本项目的 viz/ 框架，不依赖 CUDA 渲染器。

不修改上游仓库的任何文件——本脚本复用 tools/demo/demo.py 里的函数，
只是不调用它的 render_incam / render_global。

用法
----
    # 静态相机（你的素材实测是静止的）
    python run_gvhmr.py --video "processed/xxx_1280p.mp4" -s

    # 相机有运动时（走 SimpleVO，不需要 DPVO）
    python run_gvhmr.py --video "xxx.mp4"

    # 指定焦距（等效 35mm 毫米数），有助于相机标定
    python run_gvhmr.py --video "xxx.mp4" -s --f_mm 24

产物
----
    GVHMR/outputs/demo/<视频名>/
        preprocess/   检测框、2D 姿态、ViT 特征
        hmr4d_results.pt   ← 推理结果（SMPL-X 参数，含 incam 和 global 两套）

拿到 hmr4d_results.pt 后可以用 viz/ 框架可视化。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/
ROOT = HERE.parent                              # 项目根
GVHMR_ROOT = ROOT / "GVHMR"                     # 上游仓库留在根目录
SHIMS = HERE / "shims"                          # shim 随代码放在 src/ 下


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    if not GVHMR_ROOT.exists():
        print(f"找不到 GVHMR 仓库：{GVHMR_ROOT}", file=sys.stderr)
        return 1

    # shim 必须在导入 hmr4d 之前进 sys.path，否则 pytorch3d 导入会失败
    sys.path.insert(0, str(SHIMS))
    sys.path.insert(0, str(GVHMR_ROOT))
    # demo.py 内部有 "hmr4d/utils/body_model/xxx.pt" 这类相对路径，必须切到仓库根
    os.chdir(GVHMR_ROOT)

    # 6GB 显存跑 ViT-Huge 容易因显存碎片触发 OOM（HMR2 特征提取里 batch_size=16，
    # 上游注释写着"约占 5GB"）。expandable_segments 允许分配器复用碎片段，
    # 通常能显著降低峰值占用，且不需要改动上游代码。必须在 import torch 前设置。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 先确认 shim 真的生效，别等到跑到一半才发现装的是假 pytorch3d
    import pytorch3d
    if "shim" not in getattr(pytorch3d, "__version__", ""):
        print(
            "警告：导入到的不是本项目的 pytorch3d shim。\n"
            f"  实际路径：{getattr(pytorch3d, '__file__', '?')}\n"
            "  如果系统里真的装了 pytorch3d，可以直接用它，忽略本脚本。",
            file=sys.stderr,
        )

    import torch
    from tools.demo.demo import parse_args_to_cfg, run_preprocess, load_data_dict
    from hmr4d.utils.pylogger import Log
    from hmr4d.utils.net_utils import detach_to_cpu

    if not torch.cuda.is_available():
        print(
            "错误：torch 检测不到 CUDA。\n"
            "  装成 CPU 版了。重装：\n"
            "  pip install torch==2.3.0 torchvision==0.18.0 "
            "--index-url https://download.pytorch.org/whl/cu121",
            file=sys.stderr,
        )
        return 1

    # GVHMR 的 output_root 是相对它自己仓库根的（进程已 chdir 过去），
    # 而 hydra 的覆盖语法又不接受 Windows 绝对路径（冒号会被当语法符号）。
    # 这里把绝对路径改写成相对 GVHMR_ROOT 的路径，两个约束都满足。
    if "--output_root" in sys.argv:
        i = sys.argv.index("--output_root")
        if i + 1 < len(sys.argv):
            p = Path(sys.argv[i + 1])
            if p.is_absolute():
                sys.argv[i + 1] = os.path.relpath(p, GVHMR_ROOT).replace("\\", "/")

    cfg = parse_args_to_cfg()
    paths = cfg.paths

    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")

    # ===== 前置产物必须齐全，缺一个就拒绝跑 =====
    #
    # 上游 `run_preprocess` 对这三样各有一条"自己算一遍"的分支：
    #   bbx.pt 缺        -> 它自己跑 YOLO 跟踪
    #   vitpose.pt 缺    -> 它自己跑 **ViTPose-H**（2.55 GB 权重，已从盘上删掉）
    #   vit_features.pt 缺 -> 它自己跑 HMR2
    # 在本项目里这三条都是**走错路**：2D 关键点会换成 ViTPose 那份（而 export 又把
    # `vitpose.pt` 当 2D 证据读，于是导出里的关键点也不是我们测过的那套），检测框
    # 也不再是第 2 步那套。能跑完，但换了一套实现，日志里只有一句轻描淡写的
    # "VitPose fill with ..."。
    #
    # 实测这条路的收益也是负的：同一条素材上 RTMPose-x(384x288) 的胯部重投影
    # 误差比 ViTPose-H **低 28%**。既然更准的那套是我们自己的，就没必要留一条
    # 会静默换实现的退路 —— 缺产物就直接说缺什么。
    #
    # （Web 那条路本来就被 check_prereqs 挡着；这里挡的是直接敲命令行的入口。）
    missing = [p for p in (paths.bbx, paths.vitpose, paths.vit_features)
               if not Path(p).exists()]
    if missing:
        print("缺预处理产物，拒绝跑：", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print("  上游检测到缺文件会自己用**另一套模型**补（ViTPose / 自带 YOLO），"
              "那会和本项目跑出来的结果不是一套东西，所以这里直接拦下。\n"
              "  先跑第 2 步「检测框 + HMR2 特征」和第 3 步「RTMPose 2D 关键点」。",
              file=sys.stderr)
        return 1

    # ===== 预处理（检测 / 2D 姿态 / ViT 特征 / 视觉里程计）=====
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    # ===== 推理 =====
    if Path(paths.hmr4d_results).exists():
        Log.info(f"[HMR4D] 已有结果，跳过：{paths.hmr4d_results}")
    else:
        import hydra

        Log.info("[HMR4D] 开始推理")
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        Log.info(f"[HMR4D] 耗时 {Log.sync_time() - tic:.2f}s "
                 f"(视频时长 {data['length'] / 30:.1f}s)")
        torch.save(pred, paths.hmr4d_results)

    # ===== 渲染（本脚本刻意跳过）=====
    print()
    print("=" * 64)
    print("推理完成。渲染已跳过（需要 pytorch3d 的 CUDA 渲染器）。")
    print(f"结果文件：{paths.hmr4d_results}")
    print()
    try:
        pred = torch.load(paths.hmr4d_results, map_location="cpu")
        print("结果包含的键：")
        for k, v in pred.items():
            shape = getattr(v, "shape", None)
            print(f"  {k:28s} {tuple(shape) if shape else type(v).__name__}")
    except Exception as exc:
        print(f"（读取结果摘要失败：{exc}）")
    print()
    print("下一步：把结果接到 viz/ 框架做可视化与诊断。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
