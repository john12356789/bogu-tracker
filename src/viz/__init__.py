"""骨骼追踪可视化与诊断框架。

模块划分：
  schema  —— 与管线解耦的中间数据结构 MotionSequence
  sources —— 合成 demo、视频读取、各重建管线适配器
  checks  —— 诊断指标（足部穿透/足滑/重投影误差/跨方法分歧）
  app     —— Rerun 日志与蓝图布局
"""

from .schema import MotionSequence, project_points
from .sources import from_gvhmr, from_smplx_npz, from_wham, load_video_frames, make_demo
from .checks import DiagnosticReport, divergence, run_all

__all__ = [
    "MotionSequence",
    "project_points",
    "make_demo",
    "load_video_frames",
    "from_gvhmr",
    "from_wham",
    "from_smplx_npz",
    "DiagnosticReport",
    "run_all",
    "divergence",
]
