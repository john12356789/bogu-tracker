"""Meshes / join_meshes_as_scene 的桩（同 pytorch3d.renderer 的理由）。"""

from __future__ import annotations

_HINT = (
    "{name} 属于 pytorch3d 的渲染数据结构，Windows 兼容 shim 未实现。\n"
    "推理结果不依赖它——请用 viz/ 框架做可视化。"
)


def _unavailable(name: str):
    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(_HINT.format(name=name))

    _Unavailable.__name__ = name
    _Unavailable.__qualname__ = name
    return _Unavailable


Meshes = _unavailable("Meshes")
join_meshes_as_scene = _unavailable("join_meshes_as_scene")

__all__ = ["Meshes", "join_meshes_as_scene"]
