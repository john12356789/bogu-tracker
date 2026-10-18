"""数据源：合成 demo、真实视频读取、各重建管线的适配器。

合成 demo 的存在不是凑数——它刻意复现了你遇到的现象：一段骨盆被遮挡的区间，
且遮挡帧的 2D 关键点带有噪声。框架必须在 demo 上就能把「遮挡帧重投影误差升高、
可见性掩码置零」这件事显示出来，否则等模型下好了你也看不出诊断是否有效。
"""

from __future__ import annotations

import warnings

from pathlib import Path

import numpy as np

from mesh_npz import mesh_path_for

from .schema import (
    BODY_JOINT_NAMES,
    BODY_JOINT_PARENTS,
    FOOT_JOINT_IDS,
    LEFT_FOOT_JOINT_IDS,
    RIGHT_FOOT_JOINT_IDS,
    MotionSequence,
)

# 人体模型权重目录：src/viz/sources.py -> 上溯三级到项目根，再进 workspace/models。
# 用路径推导而不是硬编码相对路径，脚本从任何工作目录调用都能找到。
MODEL_ROOT = Path(__file__).resolve().parents[2] / "workspace" / "models"

# 静止姿态下各关节相对父节点的偏移（米，Y 轴向上）
REST_OFFSETS = np.array(
    [
        [0.00, 0.00, 0.00],    # 0  pelvis（根，偏移由 root_trans 给定）
        [0.09, -0.06, 0.00],   # 1  left_hip
        [-0.09, -0.06, 0.00],  # 2  right_hip
        [0.00, 0.10, 0.00],    # 3  spine1
        [0.00, -0.45, 0.00],   # 4  left_knee
        [0.00, -0.45, 0.00],   # 5  right_knee
        [0.00, 0.10, 0.00],    # 6  spine2
        [0.00, -0.45, 0.00],   # 7  left_ankle
        [0.00, -0.45, 0.00],   # 8  right_ankle
        [0.00, 0.10, 0.00],    # 9  spine3
        [0.00, -0.06, 0.12],   # 10 left_foot
        [0.00, -0.06, 0.12],   # 11 right_foot
        [0.00, 0.12, 0.00],    # 12 neck
        [0.08, 0.04, 0.00],    # 13 left_collar
        [-0.08, 0.04, 0.00],   # 14 right_collar
        [0.00, 0.12, 0.00],    # 15 head
        [0.10, 0.00, 0.00],    # 16 left_shoulder
        [-0.10, 0.00, 0.00],   # 17 right_shoulder
        [0.26, 0.00, 0.00],    # 18 left_elbow
        [-0.26, 0.00, 0.00],   # 19 right_elbow
        [0.24, 0.00, 0.00],    # 20 left_wrist
        [-0.24, 0.00, 0.00],   # 21 right_wrist
    ],
    dtype=np.float64,
)


def rodrigues(v: np.ndarray) -> np.ndarray:
    """轴角 (..., 3) -> 旋转矩阵 (..., 3, 3)。"""
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-12, 1.0, theta)
    k = v / safe
    K = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(theta)[..., None] * K + (1.0 - np.cos(theta))[..., None] * (K @ K)


def forward_kinematics(
    axis_angle: np.ndarray, root_trans: np.ndarray, offsets: np.ndarray, parents: list[int]
) -> np.ndarray:
    """正向运动学，返回世界系关节位置 (T, J, 3)。

    axis_angle: (T, J, 3)，root_trans: (T, 3)
    """
    n_frames, n_joints = root_trans.shape[0], len(parents)
    glob_R = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)
    glob_p = np.zeros((n_frames, n_joints, 3), dtype=np.float64)

    for j in range(n_joints):
        R_local = rodrigues(axis_angle[:, j])
        p = parents[j]
        if p < 0:
            glob_R[:, j] = R_local
            glob_p[:, j] = root_trans
        else:
            glob_R[:, j] = glob_R[:, p] @ R_local
            glob_p[:, j] = glob_p[:, p] + np.einsum("tij,j->ti", glob_R[:, p], offsets[j])
    return glob_p


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """构造 camera-to-world 4x4（OpenCV 相机约定：+X 右，+Y 下，+Z 前）。"""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    E = np.eye(4)
    E[:3, 0] = right
    E[:3, 1] = down
    E[:3, 2] = forward
    E[:3, 3] = eye
    return E


def make_demo(
    n_frames: int = 240,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    occlude_range: tuple[int, int] = (90, 150),
    seed: int = 0,
    model_root: str | Path | None = MODEL_ROOT,
    gender: str = "female",
) -> MotionSequence:
    """合成 demo。优先用真实 SMPL-X 模型，失败则退回火柴人。

    刻意构造的现象（两种模式一致）：
      - 左右交替屈膝抬脚 -> 足接触状态变化、可观察足滑
      - 骨盆持续摆动      -> 遮挡期间需要靠可见部位约束
      - 遮挡帧的关键点被替换为带噪声的估计值，且 visible 置 False
    """
    if model_root:
        try:
            return make_smplx_demo(
                n_frames=n_frames, fps=fps, width=width, height=height,
                occlude_range=occlude_range, seed=seed,
                model_root=model_root, gender=gender,
            )
        except Exception as exc:  # 缺模型 / 缺 torch 时静默退回
            warnings.warn(f"SMPL-X demo 不可用（{exc}），退回火柴人 demo。", stacklevel=2)
    return _make_stick_demo(
        n_frames=n_frames, fps=fps, width=width, height=height,
        occlude_range=occlude_range, seed=seed,
    )


def _attach_camera_and_evidence(
    joints: np.ndarray,
    n_joints_used: int,
    width: int,
    height: int,
    fps: float,
    occlude_range: tuple[int, int],
    seed: int,
    name: str,
    source: str,
) -> MotionSequence:
    """共用逻辑：摆相机、投影出 2D 关键点、制造遮挡掩码与噪声、判定足接触。"""
    rng = np.random.default_rng(seed)
    n_frames = joints.shape[0]

    cam_to_world = look_at(eye=(0.0, 1.20, 2.65), target=(0.0, 0.85, 0.0))
    focal = 900.0
    K = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    world_to_cam = np.linalg.inv(cam_to_world)
    cam_pts = joints @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    proj = cam_pts @ K.T
    keypoints_2d = proj[..., :2] / proj[..., 2:3]

    # 可见点加小幅检测噪声（~2px），被遮挡点加大幅噪声（~22px）。
    # 这样重投影误差诊断才有意义：可见段维持低位，遮挡段显著抬高。
    visible = np.ones((n_frames, n_joints_used), dtype=bool)
    visible[occlude_range[0] : occlude_range[1], [0, 1, 2]] = False
    noisy = keypoints_2d.copy()
    noisy += rng.normal(0.0, 2.0, size=noisy.shape)
    occl = ~visible
    noisy[occl] += rng.normal(0.0, 22.0, size=(int(occl.sum()), 2))
    conf = np.where(visible, 0.9, 0.15).astype(np.float64)

    foot_y = joints[:, FOOT_JOINT_IDS, 1]
    contact = np.stack(
        [foot_y[:, :2].min(axis=1) < 0.05, foot_y[:, 2:].min(axis=1) < 0.05], axis=1
    )

    return MotionSequence(
        name=name, fps=fps, joints=joints[:, :n_joints_used],
        K=K, cam_to_world=cam_to_world, image_size=(width, height),
        keypoints_2d=noisy, keypoints_conf=conf, keypoints_visible=visible,
        kp_to_joint=np.arange(n_joints_used), foot_contact=contact, source=source,
    )


def _dance_motion_pose(n_frames: int, fps: float) -> np.ndarray:
    """关节索引的轴角姿态 (T, 22, 3) —— 索引对应 BODY_JOINT_NAMES。"""
    t = np.arange(n_frames) / fps
    pose = np.zeros((n_frames, len(BODY_JOINT_NAMES), 3), dtype=np.float64)

    # 骨盆偏航（胯部转动，舞蹈里的关键量），脊柱反向保持上身稳定
    pose[:, 0, 1] = np.deg2rad(14) * np.sin(2 * np.pi * 0.35 * t)
    pose[:, 3, 1] = -np.deg2rad(6) * np.sin(2 * np.pi * 0.35 * t)
    pose[:, 3, 0] = np.deg2rad(4) * np.sin(2 * np.pi * 0.7 * t)

    # 左右腿交替屈膝（被屈的一侧抬脚）
    bend_l = np.deg2rad(38) * np.clip(np.sin(2 * np.pi * 0.35 * t), 0, None)
    bend_r = np.deg2rad(38) * np.clip(-np.sin(2 * np.pi * 0.35 * t), 0, None)
    pose[:, 1, 2] = -0.5 * bend_l   # left_hip
    pose[:, 4, 2] = bend_l          # left_knee
    pose[:, 2, 2] = -0.5 * bend_r   # right_hip
    pose[:, 5, 2] = bend_r          # right_knee
    pose[:, 7, 0] = 0.4 * bend_l    # left_ankle
    pose[:, 8, 0] = 0.4 * bend_r    # right_ankle

    # 手臂：SMPL-X 静止是平举 T-pose，绕 Z 转负/正角把两侧手臂放下来
    # （左臂指向 +X，θ<0 使 y 减小；右臂指向 -X，θ>0 使 y 减小）
    pose[:, 13, 2] = np.deg2rad(-10)                                   # left_collar
    pose[:, 14, 2] = np.deg2rad(10)                                    # right_collar
    pose[:, 16, 2] = np.deg2rad(-72) - 0.15 * bend_l                   # left_shoulder
    pose[:, 17, 2] = np.deg2rad(72) + 0.15 * bend_r                    # right_shoulder
    pose[:, 18, 1] = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)      # left_elbow
    pose[:, 19, 1] = -np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)     # right_elbow
    return pose


def make_smplx_demo(
    n_frames: int = 240,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    occlude_range: tuple[int, int] = (90, 150),
    seed: int = 0,
    model_root: str | Path = MODEL_ROOT,
    gender: str = "female",
) -> MotionSequence:
    """用真实 SMPL-X 模型生成 demo，输出真实拓扑的 mesh 与关节。

    模型静止姿态：双脚底 y≈-1.253，身高约 1.66 m，双臂平举。
    这里把整体抬高使双脚踩在地面 y=0。
    """
    import torch

    import smplx

    model = smplx.create(model_root, model_type="smplx", gender=gender,
                         use_pca=False, flat_hand_mean=True)
    model.eval()

    base_lift = float(-model.v_template[:, 1].min())  # 使静止姿态脚底落在 y=0
    pose = _dance_motion_pose(n_frames, fps)

    t = np.arange(n_frames) / fps
    transl = np.zeros((n_frames, 3), dtype=np.float64)
    transl[:, 0] = 0.14 * np.sin(2 * np.pi * 0.35 * t)   # 左右重心转移
    transl[:, 1] = base_lift + 0.02 * np.sin(2 * np.pi * 0.7 * t)
    transl[:, 2] = 0.05 * np.sin(2 * np.pi * 0.2 * t)

    def T(a):
        return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)

    # 面部与手部 pose 必须显式给出 batch 维度：默认值是 batch=1，
    # 与 body_pose 的 T 对不上，smplx 会在拼接 full_pose 时报尺寸错误。
    zeros3 = np.zeros((n_frames, 3))
    with torch.no_grad():
        out = model(
            betas=T(np.zeros((n_frames, 10))),
            expression=T(np.zeros((n_frames, 10))),
            global_orient=T(pose[:, 0]),
            body_pose=T(pose[:, 1:]),
            jaw_pose=T(zeros3),
            leye_pose=T(zeros3),
            reye_pose=T(zeros3),
            left_hand_pose=T(np.zeros((n_frames, 15, 3))),
            right_hand_pose=T(np.zeros((n_frames, 15, 3))),
            transl=T(transl),
        )

    vertices = out.vertices.numpy().astype(np.float64)     # (T, 10475, 3)
    joints_full = out.joints.numpy().astype(np.float64)    # (T, 127, 3)，前 22 为身体关节
    body_joints = joints_full[:, : len(BODY_JOINT_NAMES)]

    seq = _attach_camera_and_evidence(
        body_joints, len(BODY_JOINT_NAMES), width, height, fps,
        occlude_range, seed, name="demo-smplx", source=f"smplx:{gender}",
    )
    seq.vertices = vertices
    seq.faces = np.asarray(model.faces, dtype=np.uint32)
    return seq


def _make_stick_demo(
    n_frames: int = 240,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    occlude_range: tuple[int, int] = (90, 150),
    seed: int = 0,
) -> MotionSequence:
    """无模型时的火柴人回退：用简单 FK 拼出骨架，不含 mesh。"""
    n_joints = len(BODY_JOINT_PARENTS)
    t = np.arange(n_frames) / fps

    pose = _dance_motion_pose(n_frames, fps)

    root_trans = np.zeros((n_frames, 3), dtype=np.float64)
    root_trans[:, 0] = 0.14 * np.sin(2 * np.pi * 0.35 * t)
    root_trans[:, 1] = 1.02 + 0.02 * np.sin(2 * np.pi * 0.7 * t)
    root_trans[:, 2] = 0.05 * np.sin(2 * np.pi * 0.2 * t)

    joints = forward_kinematics(pose, root_trans, REST_OFFSETS, BODY_JOINT_PARENTS)

    return _attach_camera_and_evidence(
        joints, n_joints, width, height, fps, occlude_range, seed,
        name="demo-stick", source="synthetic",
    )


def render_demo_frames(seq: MotionSequence, stride: int = 1) -> list[np.ndarray]:
    """为 demo 合成假的视频帧，便于在没有真实视频时验证视频面板。"""
    width, height = seq.image_size
    frames = []
    for frame_idx in range(0, seq.n_frames, stride):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        grad = np.linspace(40, 110, height, dtype=np.uint8)
        img[:] = grad[:, None, None]

        uv = seq.keypoints_2d[frame_idx]
        vis = seq.keypoints_visible[frame_idx]
        for k in range(uv.shape[0]):
            x, y = int(round(uv[k, 0])), int(round(uv[k, 1]))
            if not (0 <= x < width and 0 <= y < height):
                continue
            color = np.array([80, 230, 120], dtype=np.uint8) if vis[k] else np.array([230, 90, 90], dtype=np.uint8)
            r = 5
            y0, y1 = max(0, y - r), min(height, y + r + 1)
            x0, x1 = max(0, x - r), min(width, x + r + 1)
            img[y0:y1, x0:x1] = color
        frames.append(img)
    return frames


def load_video_frames(
    path: str, max_frames: int | None = None, stride: int = 1
) -> tuple[list[np.ndarray], tuple[int, int], float]:
    """读取真实视频，返回 (帧列表, (宽, 高), fps)。

    优先用 imageio（依赖 ffmpeg 但轻），退回 opencv。
    """
    try:
        import imageio.v3 as iio

        meta = iio.immeta(path, plugin="pyav")
        fps = float(meta.get("fps", 30.0))
        frames = []
        for i, frame in enumerate(iio.imiter(path, plugin="pyav")):
            if i % stride:
                continue
            frames.append(np.asarray(frame))
            if max_frames is not None and len(frames) >= max_frames:
                break
        h, w = frames[0].shape[:2]
        return frames, (w, h), fps
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            i += 1
        cap.release()
        return frames, (width, height), fps
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"无法读取视频 {path}。请安装 imageio[ffmpeg] 或 opencv-python。原始错误：{exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 管线适配器
# ---------------------------------------------------------------------------
# 模型下好之后，在这里把各管线的原生输出翻译成 MotionSequence。
# 三个函数签名故意留成 **kwargs，方便你按实际输出格式填。
# 关键是填对这三件事：坐标系（cam 还是 world）、单位（米还是毫米）、关节顺序。


def from_gvhmr(**_kwargs) -> MotionSequence:
    """GVHMR -> MotionSequence。

    需要确认/填入：
      - 输出的 SMPL-X 参数是否在相机系（`verts_in_cam` 决定要不要置 True）
      - `smplx_params['trans']` 是平移，需与 `betas`/`body_pose` 一起前向得到关节位置
      - 世界系对齐后的相机外参在 `smplx_params` 还是单独文件里
      - GVHMR 的世界系是 Y 向上、重力沿 -Y（与本框架一致），但要实测确认
    """
    raise NotImplementedError(
        "GVHMR 适配器待实现：拿到输出后确认坐标系/单位/关节顺序，"
        "填出 vertices 或 joints、K、cam_to_world 即可。"
    )


def from_wham(**_kwargs) -> MotionSequence:
    """WHAM -> MotionSequence。

    注意：WHAM 输出 SMPL（不是 SMPL-X），关节顺序前 22 个与本框架一致，
    但顶点数与 SMPL-X 不同（6890 vs 10475），跨方法比较时只比关节位置。
    """
    raise NotImplementedError("WHAM 适配器待实现：同上三件事先确认。")


def from_smplx_npz(path: str, with_mesh: bool = True, **kwargs) -> MotionSequence:
    """从通用的 SMPL-X npz 结果文件读取。

    预期键（缺哪个就跳过哪个）：
      vertices (T,V,3) / joints (T,J,3) / faces (F,3)
      K (3,3)|(T,3,3) / cam_to_world (4,4)|(T,4,4) / image_size
      keypoints_2d (T,K,2) / keypoints_visible (T,K) / keypoints_conf (T,K)
      foot_contact (T,2)

    with_mesh=False 会跳过 vertices/faces。**只做姿态/关节分析时务必关掉**：
    vertices 是 10475 点 × T 帧，一条 80 秒的片子就有 307 MB，
    解压读取要 1.7 秒——而只看 joints（700 KB）根本不需要它。
    """
    data = np.load(path, allow_pickle=True)
    # 网格在**另一个文件**里（`x_incam.npz` -> `x_incam.mesh.npz`，见 mesh_npz.py）：
    # 它一个就占结果文件的 99.5%，而只有渲染/带 mesh 的分析读它，所以平时不落盘。
    # 不在就退化成"没有 mesh"，而不是报错——`keys` 那套本来就写着"缺哪个跳过哪个"。
    mesh = None
    if with_mesh:
        mp = mesh_path_for(path)
        if mp.exists():
            mesh = np.load(mp, allow_pickle=True)
        else:
            warnings.warn(f"没有网格文件 {mp.name}，只读关节；"
                          f"要网格先跑 export_gvhmr.py 补一份（20~60 秒）",
                          stacklevel=2)
    keys = ["joints", "K", "cam_to_world", "keypoints_2d",
            "keypoints_conf", "keypoints_visible", "kp_to_joint", "foot_contact"]
    if with_mesh:
        keys += ["vertices", "faces"]
    meta = {}
    for key in keys:
        src = mesh if (mesh is not None and key in mesh) else data
        if key in src:
            meta[key] = src[key]
    if "image_size" in data:
        meta["image_size"] = tuple(int(v) for v in data["image_size"])
    seq = MotionSequence(
        name=kwargs.pop("name", path),
        fps=float(kwargs.pop("fps", 30.0)),
        source=f"npz:{path}",
        **meta,
        **kwargs,
    )
    return seq.to_world()
