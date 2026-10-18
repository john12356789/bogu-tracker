"""诊断指标。

这套指标的设计目标不是"量化重建精度"（遮挡下没有真值可比），而是把**可验证的
失败模式**暴露出来：

  * 足部穿透 / 足滑 —— 世界系骨盆位置的主要约束来源，它们错了骨盆位置必然错。
    这两个量不需要真值就能判断对错，是遮挡条件下最可靠的哨兵。
  * 重投影误差（按可见性拆分）—— 关键诊断。若遮挡关节的重投影误差**同样很低**，
    说明你的管线正在拿遮挡区域的像素（道具/衣服）去凑姿态，这是最典型的坏结果；
    若遮挡关节误差合理升高、可见关节保持低误差，说明可见性掩码在起作用。
  * 跨方法分歧 —— 没有真值时的不确定度代理。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import (
    COCO17_KEYPOINT_NAMES,
    FOOT_JOINT_IDS,
    LEFT_FOOT_JOINT_IDS,
    PELVIS_REGION_JOINT_IDS,
    RIGHT_FOOT_JOINT_IDS,
    MotionSequence,
    project_points,
)


@dataclass
class DiagnosticReport:
    foot_height: np.ndarray            # (T, 2) 左右足最低点高度
    penetration: np.ndarray            # (T,) 基于关节的穿透深度
    slide: np.ndarray                  # (T,) 接触状态下足部水平位移
    jitter: np.ndarray                 # (T,) 骨盆位置二阶差分模长
    pelvis_occluded: np.ndarray | None # (T,) 骨盆区域是否不可见
    reproj: np.ndarray | None          # (T, K) 逐关节重投影误差（像素）
    reproj_visible: float | None       # 可见关节的平均重投影误差
    reproj_occluded: float | None      # 遮挡关节的平均重投影误差
    reproj_ratio: float | None         # 遮挡/可见 误差比
    reproj_vis_mask: np.ndarray | None = None   # (T, K) 可见项掩码
    reproj_occl_mask: np.ndarray | None = None  # (T, K) 遮挡项掩码
    reproj_per_kp: np.ndarray | None = None     # (K,) 每个关键点的中位误差
    mesh_penetration: np.ndarray | None = None  # (T,) 基于网格顶点的穿透深度

    def summary(self) -> str:
        lines = []
        pe = self.penetration
        if pe.size:
            lines.append(f"足部穿透     关节判据 最大 {pe.max():.4f} m，穿透帧占比 {(pe > 1e-3).mean():.1%}")
        if self.mesh_penetration is not None and self.mesh_penetration.size:
            me = self.mesh_penetration
            lines.append(f"             网格判据 最大 {me.max():.4f} m，穿透帧占比 {(me > 1e-3).mean():.1%}"
                         "   ← 这个才是「脚底有没有穿模」")
        if self.slide.size:
            lines.append(f"足滑         接触帧平均 {self.slide[self.slide > 0].mean() if (self.slide > 0).any() else 0:.4f} m/帧")
        if self.jitter.size:
            lines.append(f"骨盆抖动     均值 {self.jitter.mean():.4f} m/帧²，峰值 {self.jitter.max():.4f}")
        if self.pelvis_occluded is not None:
            lines.append(f"骨盆遮挡     占比 {self.pelvis_occluded.mean():.1%} 的帧")
        if self.reproj_visible is not None:
            lines.append(f"重投影误差   可见关节 {self.reproj_visible:.2f} px")
        if self.reproj_occluded is not None:
            ratio_txt = f"{self.reproj_ratio:.2f}" if self.reproj_ratio is not None else "n/a"
            lines.append(f"             遮挡关节 {self.reproj_occluded:.2f} px（比值 {ratio_txt}）")
            if self.reproj_ratio is not None and self.reproj_ratio < 1.3:
                lines.append(
                    "             ⚠ 遮挡关节误差没有明显升高——管线很可能在拿遮挡区域的像素凑姿态，"
                    "这不是好结果，是过拟合到了遮挡物上"
                )

        # 逐关键点拆解。汇总数会把「某个区域系统性偏差」这件事抹平，
        # 而那恰恰是最值得看的信号。
        if self.reproj_per_kp is not None:
            finite = np.isfinite(self.reproj_per_kp)
            if finite.any():
                med_all = np.nanmedian(self.reproj_per_kp)
                lines.append(f"  逐关节中位误差（整体中位 {med_all:.1f} px，只列相对整体 >1.5 倍的）:")
                idx = np.argsort(-np.nan_to_num(self.reproj_per_kp))
                shown = 0
                for i in idx:
                    e = self.reproj_per_kp[i]
                    if not np.isfinite(e) or e <= 1.5 * med_all:
                        continue
                    name = COCO17_KEYPOINT_NAMES[i] if i < len(COCO17_KEYPOINT_NAMES) else f"kp{i}"
                    mark = "  ⚠" if e > 2.0 * med_all else ""
                    lines.append(f"      {name:16s} {e:7.1f} px  ({e/med_all:.1f}×){mark}")
                    shown += 1
                    if shown >= 8:
                        break
                if shown == 0:
                    lines.append("      无——各关节误差分布均匀")
                else:
                    lines.append("      ⚠ 标记为整体 2 倍以上：这一区域的重建最不可信")
        return "\n".join(lines)


def _contact_mask(seq: MotionSequence, threshold: float) -> np.ndarray:
    """优先用管线给出的足接触；没有就按足部高度阈值推断。"""
    if seq.foot_contact is not None:
        return np.asarray(seq.foot_contact)[:, :2].astype(bool)
    foot_y = seq.joints[:, FOOT_JOINT_IDS, 1]
    return np.stack(
        [foot_y[:, :2].min(axis=1) < threshold, foot_y[:, 2:].min(axis=1) < threshold],
        axis=1,
    )


def foot_height(seq: MotionSequence) -> np.ndarray:
    """左右足最低点高度 (T, 2)。"""
    foot_y = seq.joints[:, FOOT_JOINT_IDS, 1]
    return np.stack([foot_y[:, :2].min(axis=1), foot_y[:, 2:].min(axis=1)], axis=1)


def foot_penetration(seq: MotionSequence) -> np.ndarray:
    """基于**关节**的穿透深度 (T,)，0 表示没有低于地面。

    注意这个判据会漏报：踝关节还在地面之上时，脚底网格可能已经陷下去了
    （实测差 2.9cm）。要看"渲染出来的脚有没有穿地"，用 mesh_foot_penetration。
    """
    h = foot_height(seq)
    return np.clip(seq.ground_y - h.min(axis=1), 0.0, None)


def mesh_foot_penetration(seq: MotionSequence, foot_radius: float = 0.13) -> np.ndarray | None:
    """基于**网格顶点**的穿透深度 (T,)，没有 mesh 时返回 None。

    做法：先用中位帧找出「离足部关节足够近」的顶点索引集合（网格拓扑固定，
    这套索引对所有帧都成立），再逐帧取这些顶点的最低 y。
    这才是"鞋底有没有穿模"的度量。
    """
    if seq.vertices is None or seq.joints is None:
        return None

    v, j = np.asarray(seq.vertices), np.asarray(seq.joints)
    n = min(len(v), len(j))
    mid = v[n // 2]
    foot_pts = j[n // 2][FOOT_JOINT_IDS]
    d = np.linalg.norm(mid[:, None, :] - foot_pts[None, :, :], axis=2).min(axis=1)
    idx = np.flatnonzero(d < foot_radius)
    if idx.size == 0:
        return None

    lowest = v[:n, idx, 1].min(axis=1)
    return np.clip(seq.ground_y - lowest, 0.0, None)


def foot_slide(seq: MotionSequence, contact_threshold: float = 0.05) -> np.ndarray:
    """接触状态下的足部水平位移 (T,)。世界系轨迹错误会直接在此暴露。"""
    contact = _contact_mask(seq, contact_threshold)
    n = seq.n_frames
    out = np.zeros(n, dtype=np.float64)
    for foot, ids in enumerate((LEFT_FOOT_JOINT_IDS, RIGHT_FOOT_JOINT_IDS)):
        xy = seq.joints[:, ids, [0, 2]].mean(axis=1)
        for t in range(1, n):
            if contact[t, foot] and contact[t - 1, foot]:
                out[t] += float(np.linalg.norm(xy[t] - xy[t - 1]))
    return out


def root_jitter(seq: MotionSequence) -> np.ndarray:
    """骨盆位置二阶差分模长 (T,)。前两帧为 0。"""
    root = seq.joints[:, 0, :]
    if len(root) < 3:
        return np.zeros(len(root), dtype=np.float64)
    acc = root[2:] - 2 * root[1:-1] + root[:-2]
    out = np.zeros(len(root), dtype=np.float64)
    out[2:] = np.linalg.norm(acc, axis=1)
    return out


def _joint_to_kp_map(seq: MotionSequence) -> dict[int, int]:
    """映射：3D 关节索引 -> 2D 关键点索引。

    没有显式 kp_to_joint 时才退回「同索引对应」，那**只在关键点集与关节集
    顺序一致时才成立**。COCO17 与 SMPL24 顺序完全不同（COCO 第 0 个是鼻子，
    SMPL 第 0 个是骨盆），错用会让重投影误差大出一个数量级且毫无意义——
    所以宁可不给映射，也不要猜。值为 -1 的条目表示该关键点无对应关节，跳过。
    """
    if seq.kp_to_joint is None:
        limit = min(seq.keypoints_2d.shape[1], seq.joints.shape[1])
        return {j: j for j in range(limit)}
    mapping = {}
    for kp_idx, joint_idx in enumerate(np.asarray(seq.kp_to_joint).ravel()):
        if int(joint_idx) < 0:
            continue
        mapping[int(joint_idx)] = kp_idx
    return mapping


def reprojection_error(seq: MotionSequence) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """逐关节重投影误差。

    返回 (reproj, visible_mask, occluded_mask)，形状均为 (T, K)，reproj 单位像素。
    任一项为 None 表示数据不足（缺关键点或缺相机参数）。
    """
    if seq.keypoints_2d is None or seq.K is None or seq.cam_to_world is None:
        return None, None, None

    n_frames = seq.n_frames
    n_kp = seq.keypoints_2d.shape[1]
    reproj = np.full((n_frames, n_kp), np.nan, dtype=np.float64)

    mapping = _joint_to_kp_map(seq)
    for t in range(n_frames):
        K = seq.k_matrix(t)
        w2c = seq.world_to_cam(t)
        if K is None or w2c is None:
            continue
        joint_ids = sorted(mapping)
        pts = seq.joints[t, joint_ids, :]
        uv, z = project_points(pts, K, w2c)
        for row, joint_idx in enumerate(joint_ids):
            if z[row] <= 1e-6:
                continue
            kp_idx = mapping[joint_idx]
            reproj[t, kp_idx] = np.linalg.norm(uv[row] - seq.keypoints_2d[t, kp_idx])

    if seq.keypoints_visible is not None:
        vis = np.asarray(seq.keypoints_visible).astype(bool)
        occl = ~vis
    else:
        vis = ~np.isnan(reproj)
        occl = np.zeros_like(vis, dtype=bool)

    vis = vis & ~np.isnan(reproj)
    occl = occl & ~np.isnan(reproj)
    return reproj, vis, occl


def divergence(seq_a: MotionSequence, seq_b: MotionSequence) -> np.ndarray:
    """两个方法在同一段视频上的逐帧关节平均距离 (T,)，单位米。

    跨方法比较**只比关节位置**——不同身体模型的 shape space 不兼容，
    直接比较参数没有意义。
    """
    n = min(seq_a.n_frames, seq_b.n_frames)
    j = min(seq_a.joints.shape[1], seq_b.joints.shape[1])
    a = seq_a.joints[:n, :j, :]
    b = seq_b.joints[:n, :j, :]
    return np.linalg.norm(a - b, axis=2).mean(axis=1)


def run_all(
    seq: MotionSequence,
    other: MotionSequence | None = None,
    contact_threshold: float = 0.05,
) -> tuple[DiagnosticReport, np.ndarray | None]:
    """跑全部诊断。返回 (报告, 跨方法分歧或 None)。"""
    if seq.joints is None:
        raise ValueError("诊断需要 joints；只有 mesh 的话请先用 J_regressor 回归关节")

    seq = seq.to_world()

    pelvis_occluded = None
    if seq.keypoints_visible is not None:
        vis = np.asarray(seq.keypoints_visible).astype(bool)
        cols = [p for p in PELVIS_REGION_JOINT_IDS if p < vis.shape[1]]
        if cols:
            pelvis_occluded = ~vis[:, cols].any(axis=1)

    reproj, vis_mask, occl_mask = reprojection_error(seq)
    rv = ro = ratio = None
    per_kp = None
    if reproj is not None and vis_mask is not None:
        if vis_mask.any():
            rv = float(np.nanmean(reproj[vis_mask]))
        if occl_mask is not None and occl_mask.any():
            ro = float(np.nanmean(reproj[occl_mask]))
        if rv and ro and rv > 1e-9:
            ratio = ro / rv
    if reproj is not None:
        # 未被映射的关键点整列为 NaN，nanmedian 会发 All-NaN 警告，这里显式静音
        import warnings as _w

        with np.errstate(all="ignore"), _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            per_kp = np.nanmedian(reproj, axis=0)  # (K,) 每个关键点沿时间的中位误差

    report = DiagnosticReport(
        foot_height=foot_height(seq),
        penetration=foot_penetration(seq),
        mesh_penetration=mesh_foot_penetration(seq),
        slide=foot_slide(seq, contact_threshold),
        jitter=root_jitter(seq),
        pelvis_occluded=pelvis_occluded,
        reproj=reproj,
        reproj_visible=rv,
        reproj_occluded=ro,
        reproj_ratio=ratio,
        reproj_vis_mask=vis_mask,
        reproj_occl_mask=occl_mask,
        reproj_per_kp=per_kp,
    )
    div = divergence(seq, other) if other is not None else None
    return report, div
