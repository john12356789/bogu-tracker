"""pytorch3d.structures 的桩。理由同 pytorch3d.renderer：仅用于出图，推理不需要。"""

from __future__ import annotations

from .meshes import Meshes, join_meshes_as_scene  # noqa: F401

__all__ = ["Meshes", "join_meshes_as_scene"]
