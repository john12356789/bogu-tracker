"""网格（vertices）与重建结果分开落盘时，那个文件的命名规则。

**只有这一份。** 用它的人横跨两个 Python 环境、读写两端：

    写入端   export_gvhmr.py        （gvhmr 环境，3.10 + torch）
    读取端   render_overlay.py      （主环境，3.12）
    读取端   viz/sources.py         （同上，MotionSequence 那条路）
    补网格   batch_run.py           （渲染前发现网格不在就重跑导出）
    清理     strip_mesh.py          （历史产物去网格）

规则抄第二遍就会漂，而漂的表现是"文件明明在，却报找不到"——所以这里只放一行
变换，谁都能 import，谁都不需要依赖别的模块。
"""
from __future__ import annotations

from pathlib import Path

MESH_SUFFIX = ".mesh.npz"


def mesh_path_for(npz: str | Path) -> Path:
    """`x_incam.npz` -> `x_incam.mesh.npz`（不是 npz 就原样加后缀）。

    为什么网格要单独一个文件：它一个就占原 npz 的 99.5%（10475 顶点 × 3 ×
    float32 = 每帧 123 KB），而**只有渲染和 viz 读它**——3D 视图、算脚本、前端的
    motion 接口都只用 joints（每帧 300 字节）和 2D 关键点。捆在一起的话，一条
    从不渲染的素材也要一直占着几百 MB；拆开之后"不渲染就不占"才成立，
    要渲染时用同一份 hmr4d_results.pt 重跑一次导出（20~60 秒）就长回来。
    """
    p = Path(npz)
    if p.name.endswith(".npz"):
        return p.with_name(p.name[: -len(".npz")] + MESH_SUFFIX)
    return p.with_name(p.name + MESH_SUFFIX)
