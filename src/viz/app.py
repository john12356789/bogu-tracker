"""Rerun 可视化：3D 场景 + 视频对齐面板 + 诊断时间序列。

布局设计是刻意的——三个面板必须能同时看：
  * 3D 视图：mesh / 骨架 / 地面 / 相机。**注意这个视图在遮挡段会"看起来很对"**，
    它验证不了你的问题，别盯着它下结论。
  * 视频面板：合成帧或真实视频，关键点按可见性着色（绿=可见，红=遮挡）。
  * 时间序列：骨盆高度、足部高度、重投影误差、跨方法分歧。**这里才是暴露问题的地方。**
"""

from __future__ import annotations

import warnings

import numpy as np

from .checks import DiagnosticReport
from .schema import (
    BODY_BONES,
    FOOT_JOINT_IDS,
    PELVIS_REGION_JOINT_IDS,
    MotionSequence,
)

_rerun_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _rerun_warned:
        _rerun_warned.add(key)
        warnings.warn(message, stacklevel=2)


# ---------------------------------------------------------------------------
# Rerun API 兼容层
# ---------------------------------------------------------------------------
def set_time(rr, frame_idx: int, seconds: float) -> None:
    """时间轴。Rerun 0.20 前后 API 变了，两个都试。"""
    try:
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", duration=seconds)
        return
    except (TypeError, AttributeError):
        pass
    try:
        rr.set_time_sequence("frame", frame_idx)
        rr.set_time_seconds("time", seconds)
        return
    except (TypeError, AttributeError):
        _warn_once("set_time", "无法设置时间轴，各版本 API 都不匹配；帧将全部落在同一时刻。")


def log_scalar(rr, path: str, value: float) -> None:
    try:
        rr.log(path, rr.Scalars([float(value)]))
    except AttributeError:
        rr.log(path, rr.Scalar(float(value)))


def log_mesh(rr, path: str, vertices: np.ndarray, faces: np.ndarray,
             vertex_colors: np.ndarray | None = None) -> None:
    kwargs = dict(vertex_positions=np.asarray(vertices, dtype=np.float32))
    if vertex_colors is not None:
        kwargs["vertex_colors"] = np.asarray(vertex_colors, dtype=np.uint8)
    try:
        rr.log(path, rr.Mesh3D(triangle_indices=np.asarray(faces, dtype=np.uint32), **kwargs))
    except TypeError:
        rr.log(path, rr.Mesh3D(indices=np.asarray(faces, dtype=np.uint32), **kwargs))


def log_transform(rr, path: str, mat: np.ndarray) -> None:
    translation = mat[:3, 3].astype(np.float32)
    try:
        rr.log(path, rr.Transform3D(translation=translation, mat3x3=mat[:3, :3].astype(np.float32)))
    except TypeError:
        try:
            q = _mat_to_quat(mat[:3, :3])
            rr.log(path, rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=q)))
        except Exception:
            _warn_once("transform", "Transform3D API 不匹配，相机位姿未记录。3D 视图里相机可能不在正确位置。")


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 四元数 (x, y, z, w)。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


# ---------------------------------------------------------------------------
# 场景构建
# ---------------------------------------------------------------------------
def _ground_grid(ground_y: float, extent: float = 2.0, step: float = 0.5) -> list[np.ndarray]:
    coords = np.arange(-extent, extent + 1e-6, step)
    strips = []
    for c in coords:
        strips.append(np.array([[c, ground_y, -extent], [c, ground_y, extent]], dtype=np.float32))
        strips.append(np.array([[-extent, ground_y, c], [extent, ground_y, c]], dtype=np.float32))
    return strips


def _pelvis_vertex_colors(vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """按到骨盆区域关节的距离给顶点上色，把感兴趣区域在 mesh 上标出来。"""
    pelvis = joints[PELVIS_REGION_JOINT_IDS].mean(axis=0)
    d = np.linalg.norm(vertices - pelvis, axis=1)
    scale = np.clip(d / 0.45, 0.0, 1.0)
    hot = np.array([255, 80, 60], dtype=np.float32)
    cold = np.array([140, 150, 165], dtype=np.float32)
    return (cold + (hot - cold) * (1.0 - scale)[:, None]).astype(np.uint8)


def log_sequence(
    rr,
    seq: MotionSequence,
    report: DiagnosticReport,
    video_frames: list[np.ndarray] | None = None,
    video_stride: int = 1,
    divergence: np.ndarray | None = None,
    ghost: MotionSequence | None = None,
    world_root: str = "world",
) -> None:
    """把一条 MotionSequence 及其诊断结果写进 Rerun。"""
    seq = seq.to_world()
    n = seq.n_frames
    fps = seq.fps or 30.0

    rr.log(f"{world_root}/ground", rr.LineStrips3D(
        _ground_grid(seq.ground_y), colors=np.array([90, 90, 100], dtype=np.uint8), radii=0.0015))

    if seq.has_mesh:
        log_mesh(rr, f"{world_root}/mesh", seq.vertices[0], seq.faces,
                 _pelvis_vertex_colors(seq.vertices[0], seq.joints[0]))

    if ghost is not None and ghost.joints is not None:
        ghost = ghost.to_world()

    video_index = 0
    for t in range(n):
        set_time(rr, t, t / fps)

        if seq.has_mesh:
            log_mesh(rr, f"{world_root}/mesh", seq.vertices[t], seq.faces,
                     _pelvis_vertex_colors(seq.vertices[t], seq.joints[t]))

        if seq.joints is not None:
            pts = seq.joints[t]
            radii = np.full(len(pts), 0.012, dtype=np.float32)
            radii[PELVIS_REGION_JOINT_IDS] = 0.030
            if seq.keypoints_visible is not None and t < len(seq.keypoints_visible):
                vis = np.asarray(seq.keypoints_visible)[t]
                colors = np.where(vis[: len(pts), None],
                                  np.array([80, 230, 120], dtype=np.uint8),
                                  np.array([230, 70, 70], dtype=np.uint8))
            else:
                colors = np.tile(np.array([80, 230, 120], dtype=np.uint8), (len(pts), 1))
            rr.log(f"{world_root}/joints", rr.Points3D(pts.astype(np.float32), colors=colors, radii=radii))
            rr.log(f"{world_root}/bones", rr.LineStrips3D(
                [[pts[a], pts[b]] for a, b in BODY_BONES],
                colors=np.array([120, 190, 255], dtype=np.uint8), radii=0.006))
            # 骨盆轨迹拖尾：世界系轨迹错误在这里一眼可见
            rr.log(f"{world_root}/pelvis_trail", rr.LineStrips3D(
                [seq.joints[: t + 1, 0, :].astype(np.float32)],
                colors=np.array([255, 200, 60], dtype=np.uint8), radii=0.004))

        if ghost is not None and ghost.joints is not None and t < ghost.n_frames:
            gp = ghost.joints[t]
            rr.log(f"{world_root}/ghost_joints", rr.Points3D(
                gp.astype(np.float32),
                colors=np.tile(np.array([255, 130, 240], dtype=np.uint8), (len(gp), 1)),
                radii=np.full(len(gp), 0.010, dtype=np.float32)))

        if seq.extrinsics(t) is not None:
            K = seq.k_matrix(t)
            if K is not None and seq.image_size is not None:
                rr.log(f"{world_root}/camera", rr.Pinhole(
                    image_from_camera=K.astype(np.float32),
                    resolution=[int(seq.image_size[0]), int(seq.image_size[1])]))
            log_transform(rr, f"{world_root}/camera", seq.extrinsics(t))

        if video_frames and t % video_stride == 0 and video_index < len(video_frames):
            rr.log("video", rr.Image(video_frames[video_index]))
            video_index += 1

        # ---- 诊断指标（实体路径保持 ASCII，Rerun 的路径是标识符）----
        log_scalar(rr, "metrics/pelvis_height", seq.joints[t, 0, 1])
        log_scalar(rr, "metrics/pelvis_jitter", report.jitter[t])
        log_scalar(rr, "metrics/foot_height_L", report.foot_height[t, 0])
        log_scalar(rr, "metrics/foot_height_R", report.foot_height[t, 1])
        log_scalar(rr, "metrics/foot_penetration", report.penetration[t])
        log_scalar(rr, "metrics/foot_slide", report.slide[t])
        if report.reproj is not None:
            log_scalar(rr, "metrics/reproj_visible",
                       _masked_mean(report.reproj[t], report.reproj_vis_mask[t] if report.reproj_vis_mask is not None else None))
            log_scalar(rr, "metrics/reproj_occluded",
                       _masked_mean(report.reproj[t], report.reproj_occl_mask[t] if report.reproj_occl_mask is not None else None))
        if report.pelvis_occluded is not None:
            log_scalar(rr, "metrics/pelvis_occluded", float(report.pelvis_occluded[t]))
        if divergence is not None and t < len(divergence):
            log_scalar(rr, "metrics/divergence", divergence[t])


def _masked_mean(row: np.ndarray, mask: np.ndarray | None) -> float:
    """在给定掩码内求均值；掩码为空则该帧记 NaN（时间序列上会断开，正是我们想看到的）。"""
    if mask is None:
        return float("nan")
    sel = np.asarray(mask, dtype=bool) & np.isfinite(row)
    return float(np.mean(row[sel])) if sel.any() else float("nan")


# ---------------------------------------------------------------------------
# 蓝图
# ---------------------------------------------------------------------------
def send_blueprint(rr, has_video: bool, has_metrics: bool = True, has_ghost: bool = False) -> None:
    try:
        import rerun.blueprint as rrb
    except ImportError:
        _warn_once("blueprint", "找不到 rerun.blueprint，跳过布局设置，用默认布局。")
        return

    right = []
    if has_video:
        right.append(rrb.Spatial2DView(origin="video", name="视频 / 2D 关键点"))
    if has_metrics:
        right.append(rrb.TimeSeriesView(origin="metrics", name="诊断指标"))

    views = [rrb.Spatial3DView(origin="world", name="3D 场景（遮挡段会看起来'很对'，别在这下结论）")]
    if right:
        views.append(rrb.Vertical(*right, row_shares=[1, 2]))

    try:
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views, column_shares=[2, 1]), collapse_panels=True))
    except Exception as exc:
        _warn_once("send_blueprint", f"发送蓝图失败（{exc}），用默认布局。")
