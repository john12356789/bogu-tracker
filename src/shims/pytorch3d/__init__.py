"""pytorch3d 的 Windows 兼容替代（仅实现 GVHMR 推理路径所需部分）。

背景：Meta 未发布 Windows wheel，而 GVHMR 的 requirements.txt 锁死 Linux 直链。
本包用纯 PyTorch 实现 transforms，并提供 knn_points；CUDA 渲染器部分留桩。

被替代的只有 pytorch3d.transforms 和 pytorch3d.ops.knn —— 这两者都是纯数学。
真正的渲染器（pytorch3d.renderer / structures）在本包里是桩，一旦被调用会明确报错，
而不是静默产出错误结果。
"""

__version__ = "0.7.6+win-shim"

from . import transforms  # noqa: F401
