"""与重建管线解耦的中间数据结构。

任何管线（GVHMR / WHAM / SLAHMR / SMPLify-X ...）只要能填出 MotionSequence，
就能接进这套可视化与诊断框架。

约定
----
世界坐标系：右手系，Y 轴向上，重力方向 -Y，单位米。
相机坐标系：OpenCV 约定，+X 右，+Y 下，+Z 朝前（看向场景）。
外参以 camera-to-world 存储（4x4）。
所有 *_in_cam 为 True 的字段表示数据仍在相机系下，可视化前需用外参变换。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# SMPL-X / SMPL 前 22 个身体关节的顺序（两者一致），索引即骨骼树顺序
BODY_JOINT_NAMES = [
    "pelvis",         # 0  根节点，你关心的对象
    "left_hip",       # 1
    "right_hip",      # 2
    "spine1",         # 3
    "left_knee",      # 4
    "right_knee",     # 5
    "spine2",         # 6
    "left_ankle",     # 7
    "right_ankle",    # 8
    "spine3",         # 9
    "left_foot",      # 10
    "right_foot",     # 11
    "neck",           # 12
    "left_collar",    # 13
    "right_collar",   # 14
    "head",           # 15
    "left_shoulder",  # 16
    "right_shoulder", # 17
    "left_elbow",     # 18
    "right_elbow",    # 19
    "left_wrist",     # 20
    "right_wrist",    # 21
]

# SMPL-X 身体骨骼树的父节点索引，-1 表示根
BODY_JOINT_PARENTS = [
    -1,  # pelvis
    0, 0,  # left_hip, right_hip
    0,     # spine1
    1, 2,  # left_knee, right_knee
    3,     # spine2
    4, 5,  # left_ankle, right_ankle
    6,     # spine3
    7, 8,  # left_foot, right_foot
    9,     # neck
    9, 9,  # left_collar, right_collar
    12,    # head
    13, 14,  # left_shoulder, right_shoulder
    16, 17,  # left_elbow, right_elbow
    18, 19,  # left_wrist, right_wrist
]

# 骨骼连线，给 3D 视图画骨架用
BODY_BONES = [
    (p, c) for c, p in enumerate(BODY_JOINT_PARENTS) if p >= 0
]

# 用于足部接触/穿透分析的关节
FOOT_JOINT_IDS = [7, 8, 10, 11]  # 左右踝 + 左右脚
LEFT_FOOT_JOINT_IDS = [7, 10]
RIGHT_FOOT_JOINT_IDS = [8, 11]

# 骨盆区域关节——遮挡分析的核心对象
PELVIS_REGION_JOINT_IDS = [0, 1, 2]

# COCO17 关键点顺序。重投影误差按这个顺序报出来才有可读性——
# 只看一个汇总数会掩盖"某个区域系统性地差"这种最重要的信号。
COCO17_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class MotionSequence:
    """一段时间对齐的人体运动数据。各字段除 name/fps 外均可为 None。"""

    name: str = "seq"
    fps: float = 30.0

    # ---- 几何（世界系，米）----
    vertices: np.ndarray | None = None      # (T, V, 3)
    faces: np.ndarray | None = None         # (F, 3)，与时间无关
    joints: np.ndarray | None = None        # (T, J, 3)
    joint_names: list[str] = field(default_factory=lambda: list(BODY_JOINT_NAMES))

    # ---- 相机 ----
    K: np.ndarray | None = None             # (3, 3) 或 (T, 3, 3)
    cam_to_world: np.ndarray | None = None  # (4, 4) 或 (T, 4, 4)
    image_size: tuple[int, int] | None = None  # (宽, 高)

    # ---- 2D 证据 ----
    keypoints_2d: np.ndarray | None = None      # (T, K, 2) 像素坐标
    keypoints_conf: np.ndarray | None = None    # (T, K) 置信度 0..1
    keypoints_visible: np.ndarray | None = None # (T, K) bool，遮挡掩码
    # 2D 关键点索引 -> 3D 关节索引的映射；None 表示前 min(K, J) 个一一对应
    kp_to_joint: np.ndarray | None = None

    # ---- 接触 ----
    foot_contact: np.ndarray | None = None  # (T, 2) bool，[左脚, 右脚]

    # ---- 元信息 ----
    ground_y: float = 0.0
    verts_in_cam: bool = False   # vertices 是否仍在相机系
    joints_in_cam: bool = False  # joints 是否仍在相机系
    source: str = ""

    def __post_init__(self) -> None:
        if self.vertices is not None:
            self.n_frames = len(self.vertices)
        elif self.joints is not None:
            self.n_frames = len(self.joints)
        elif self.keypoints_2d is not None:
            self.n_frames = len(self.keypoints_2d)
        else:
            raise ValueError("MotionSequence 至少要提供 vertices / joints / keypoints_2d 之一")

    @property
    def has_mesh(self) -> bool:
        return self.vertices is not None and self.faces is not None

    def k_matrix(self, t: int) -> np.ndarray | None:
        """取第 t 帧的内参矩阵。"""
        if self.K is None:
            return None
        K = np.asarray(self.K)
        return K[t] if K.ndim == 3 else K

    def extrinsics(self, t: int) -> np.ndarray | None:
        """取第 t 帧的 camera-to-world 4x4 矩阵。"""
        if self.cam_to_world is None:
            return None
        E = np.asarray(self.cam_to_world)
        return E[t] if E.ndim == 3 else E

    def world_to_cam(self, t: int) -> np.ndarray | None:
        E = self.extrinsics(t)
        return None if E is None else np.linalg.inv(E)

    def to_world(self) -> "MotionSequence":
        """把仍在相机系下的顶点/关节变换到世界系，返回新对象（不改原对象）。"""
        if not (self.verts_in_cam or self.joints_in_cam) or self.cam_to_world is None:
            return self
        out = MotionSequence(
            name=self.name, fps=self.fps, faces=self.faces, K=self.K,
            cam_to_world=self.cam_to_world, image_size=self.image_size,
            keypoints_2d=self.keypoints_2d, keypoints_conf=self.keypoints_conf,
            keypoints_visible=self.keypoints_visible, kp_to_joint=self.kp_to_joint,
            foot_contact=self.foot_contact, ground_y=self.ground_y,
            joint_names=self.joint_names, source=self.source,
            vertices=self.vertices, joints=self.joints,
        )
        n = self.n_frames
        for attr, flag in (("vertices", self.verts_in_cam), ("joints", self.joints_in_cam)):
            arr = getattr(out, attr)
            if arr is None or not flag:
                continue
            moved = np.empty_like(arr)
            for t in range(n):
                E = self.extrinsics(t)
                pts = arr[t] @ E[:3, :3].T + E[:3, 3]
                moved[t] = pts
            setattr(out, attr, moved)
        out.verts_in_cam = False
        out.joints_in_cam = False
        return out


def project_points(points: np.ndarray, K: np.ndarray, world_to_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把世界系点投到像素平面。

    返回 (uv, z)，uv 为 (N, 2) 像素坐标，z 为相机系深度（用于剔除背后点）。
    """
    pts = np.asarray(points, dtype=np.float64)
    R, t = world_to_cam[:3, :3], world_to_cam[:3, 3]
    cam = pts @ R.T + t
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv_h = (K @ cam.T).T
    uv = uv_h[:, :2] / safe_z[:, None]
    return uv, z
