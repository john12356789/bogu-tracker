"""pytorch3d.renderer 的桩。

真正的渲染器依赖 CUDA 编译的扩展，没有 Windows 版。但 GVHMR 的渲染**只用于出图**——
推理结果先存盘，渲染是之后的独立步骤。所以我们不实现它，只在被真正调用时明确报错。

这样设计是刻意的：如果沉默地返回假数据，会产出"看起来对"的可视化结果，
反而掩盖问题。明确报错逼着调用方走我们自己的 viz 框架。
"""

from __future__ import annotations

_RENDERER_HINT = (
    "{name} 需要 pytorch3d 的 CUDA 渲染器，本项目用的是 Windows 兼容 shim，未实现。\n"
    "渲染在 GVHMR 里只用于生成结果视频，推理数据已存盘（hmr4d_results.pt）。\n"
    "可视化请用本项目的 viz/ 框架：\n"
    "    python run_viz.py --npz <结果文件>\n"
    "或跳过渲染直接取数据。"
)


def _unavailable(name: str):
    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(_RENDERER_HINT.format(name=name))

    _Unavailable.__name__ = name
    _Unavailable.__qualname__ = name
    return _Unavailable


PerspectiveCameras = _unavailable("PerspectiveCameras")
TexturesVertex = _unavailable("TexturesVertex")
PointLights = _unavailable("PointLights")
Materials = _unavailable("Materials")
RasterizationSettings = _unavailable("RasterizationSettings")
MeshRenderer = _unavailable("MeshRenderer")
MeshRasterizer = _unavailable("MeshRasterizer")
SoftPhongShader = _unavailable("SoftPhongShader")
FoVOrthographicCameras = _unavailable("FoVOrthographicCameras")
AmbientLights = _unavailable("AmbientLights")

from .cameras import look_at_rotation  # noqa: E402,F401

__all__ = [
    "PerspectiveCameras", "TexturesVertex", "PointLights", "Materials",
    "RasterizationSettings", "MeshRenderer", "MeshRasterizer", "SoftPhongShader",
    "FoVOrthographicCameras", "AmbientLights", "look_at_rotation",
]
