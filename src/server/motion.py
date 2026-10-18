"""npz -> 前端契约。

复用 src/viz/schema.py 的约定，不另立一套：
  · 关节名与骨骼树来自 schema.BODY_JOINT_NAMES / BODY_BONES（24 点 SMPL 拓扑）
  · 2D 关键点名来自 schema.COCO17_KEYPOINT_NAMES（RTMPose 出的就是 COCO-17）
  · 世界系 Y 轴向上、米制；相机系 OpenCV 约定

为什么要显式给 `kp_names` / `joint_names` / `bones`：
FSP 的前端是靠"可见点 > 20 个就猜是 33 点"这种启发式来区分拓扑的，
因为它的数据层没有显式标记。我们这边**永远是 COCO-17 + 24 关节**，
所以直接把拓扑发过去，让前端不必猜——顺手把那处隐患修掉。

体积：24 关节 + 17 关键点 ≈ 0.13k 数字/帧，4 位小数后约 1.1 KB/帧。
一条 80 秒的片子约 2.7 MB。走 localhost 可接受。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from . import paths

# 关节名/骨骼树从 viz.schema 取，避免两处定义漂移
from viz.schema import BODY_BONES, BODY_JOINT_NAMES, COCO17_KEYPOINT_NAMES  # noqa: E402

_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], dict] = {}

# 关键点索引 -> 3D 关节索引的默认映射（COCO-17 的髋/膝/踝对应 SMPL 的哪些关节）。
# npz 里自带 kp_to_joint，优先用它；这个表只是它缺失时的兜底。
_KP_FALLBACK = {
    5: 16, 6: 17,      # 肩
    7: 18, 8: 19,      # 肘
    9: 20, 10: 21,     # 腕
    11: 1, 12: 2,      # 髋
    13: 4, 14: 5,      # 膝
    15: 7, 16: 8,      # 踝
}


def _npz_path(slug: str, space: str, tid: int | None = None) -> Path | None:
    """`tid=None` 是主轨迹；给 tid 就是那条轨迹自己的 npz（多人素材）。"""
    d = paths.results_dir(slug, tid)
    if d is None:
        return None
    name = "hmr4d_results_incam.npz" if space == "incam" else "hmr4d_results_global.npz"
    p = d / name
    return p if p.exists() else None


def load(slug: str, space: str = "global", tid: int | None = None) -> dict | None:
    """读 npz 并常驻缓存。

    只留要用的数组：joints 700 KB、keypoints_2d 330 KB。
    **绝不缓存 vertices**——那是 307 MB，funscript 和 UI 都用不到它，
    而 CLI 那 2 秒的开销有一大半就花在读它上面。

    关于两个坐标系的歧义（踩过一次，记在这里）
    ------------------------------------------
    `global` 和 `incam` **不是同一个世界系的两个视角，而是两套 Y 轴约定**：

      global  重力对齐，**+Y 朝上**，脚在 y≈0。3D 视图该用这个。
      incam   OpenCV 相机系，**+Y 朝下**。3D 视图直接用会上下颠倒。

    而且 `incam` 经 `MotionSequence.to_world()` 变换后**并不等于 global**
    （实测逐点最大差 2.67 m）——导出到 incam 的 `cam_to_world` 里不含重力对齐，
    它的旋转保留了相机系的 Y。所以 to_funscript 那边刻意不依赖世界轴标签，
    基向量全部由关节导出（见 to_funscript.pelvis_frames 与姿态自检）。

    参数值刻意跟 npz 文件名一致，不再叫 "world"——那正是歧义的来源。
    """
    space = "incam" if space == "incam" else "global"
    key = (slug, space, tid)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    # tid 必须传下去。漏掉的后果不是报错，是**选谁都给你主轨迹的数据**，
    # 而且因为缓存键带了 tid，那份错误数据还会一直被复用
    path = _npz_path(slug, space, tid)
    if path is None:
        return None
    try:
        z = np.load(path)
        data = {
            "path": path,
            "joints": np.asarray(z["joints"], dtype=np.float32),
            "keypoints_2d": np.asarray(z["keypoints_2d"], dtype=np.float32)
            if "keypoints_2d" in z.files else None,
            "keypoints_conf": np.asarray(z["keypoints_conf"], dtype=np.float32)
            if "keypoints_conf" in z.files else None,
            "kp_to_joint": np.asarray(z["kp_to_joint"], dtype=np.int32)
            if "kp_to_joint" in z.files else None,
            "fps": float(z["fps"][0]) if "fps" in z.files else 30.0,
            "image_size": [int(v) for v in z["image_size"]] if "image_size" in z.files else None,
            "K": np.asarray(z["K"], dtype=np.float32) if "K" in z.files else None,
        }
    except Exception:                              # noqa: BLE001
        return None

    with _LOCK:
        _CACHE[key] = data
    return data


def describe(slug: str, space: str = "global", tid: int | None = None,
             lite: bool = False) -> dict:
    """给前端的元信息 + 全部逐帧数据。`tid` 选哪条轨迹（多人素材）。

    `lite=True` 只回**元信息**（帧数 / 帧率 / 画面尺寸 / 名字表），`frames` 是空的。
    为什么需要它：完整那份是一条 2.5 MB 的 JSON、服务端首次数百毫秒，而**默认
    状态下前端一帧都用不上**——2D 骨骼叠加默认关、3D 视图和骨盆轨迹默认关。
    可 videoMeta（时间轴的时长、帧号、fps）偏偏是从这里推出来的，所以"打开一条
    素材"又必须拿到那几个数。于是这一步走 lite，真正的逐帧数据等用户真打开某个
    视图时再取（见前端 VideoList 的 ensureFullMotion）。
    """
    d = load(slug, space, tid)
    if d is None:
        return {"ok": False, "error": f"没有可用的 npz（space={space}"
                                      + (f", track={tid}" if tid is not None else "") + "）"}

    joints = d["joints"]                            # (T, 24, 3) 米
    kp = d["keypoints_2d"]                          # (T, 17, 2) 像素
    conf = d["keypoints_conf"]
    T = int(joints.shape[0])
    W, H = (d["image_size"] or [0, 0])

    kp_to_joint = (d["kp_to_joint"] if d["kp_to_joint"] is not None
                   else np.array([_KP_FALLBACK.get(i, -1) for i in range(17)], dtype=np.int32))

    # 逐帧数据在这里就分岔：lite 直接返回**同一套头字段**（这里和下面那份
    # return 是同一段代码的两个出口，不是两份表——两份表迟早会漂）。
    if lite:
        # `frames` 要给**空数组**而不是不给：前端那份类型要求它存在，而且
        # "frames.length === 0" 正是前端判断"这份是 lite、要用完整的"的判据。
        return _head(slug, space, tid, d, T, W, H, kp_to_joint) | {"frames": []}

    # 必须先转 float64 再 round：对 float32 做 round 再 tolist()，
    # 会把 0.0084 展开成 0.00839999970048666（19 字符），JSON 白白大三倍。
    j_out = np.round(joints.astype(np.float64), 4)
    kp64 = kp.astype(np.float64) if kp is not None else None
    frames = []
    for t in range(T):
        item: dict = {"t": t, "j": j_out[t].tolist()}
        if kp64 is not None and t < len(kp64):
            # 像素 -> 归一化 0..1：FSP 的 canvas 是按归一化坐标画的
            pts = kp64[t].copy()
            if W and H:
                pts[:, 0] /= W
                pts[:, 1] /= H
            c = conf[t] if conf is not None and t < len(conf) else np.ones(len(pts), np.float32)
            item["k"] = np.round(
                np.concatenate([pts, np.asarray(c, np.float64)[:, None]], axis=1), 4
            ).tolist()
        frames.append(item)

    return _head(slug, space, tid, d, T, W, H, kp_to_joint) | {"frames": frames}


def _head(slug: str, space: str, tid: int | None, d: dict, T: int,
          W: int, H: int, kp_to_joint) -> dict:
    """两份返回共用的头字段（lite 和完整版各加自己的那一块）。

    抽出来是因为这些字段**前端每一条都必须有**（n_frames/fps 推时间轴，
    kp_to_joint 和骨架拓扑画 3D）。抄成两份的话，改一处忘一处就会在 lite
    那条路上悄悄少一个字段，而报出来的错离原因很远。
    """
    return {
        "ok": True,
        "slug": slug,
        "space": space,
        "track": tid,
        "n_frames": T,
        "fps": d["fps"],
        "duration": T / d["fps"] if d["fps"] else None,
        "image_size": [W, H],
        "joint_names": BODY_JOINT_NAMES,             # 22 个；joints 有 24 个，后 2 个是手
        "kp_names": COCO17_KEYPOINT_NAMES,
        "bones": [list(b) for b in BODY_BONES],
        "kp_to_joint": kp_to_joint.tolist(),
    }


# describe() 的**序列化结果**，按 (slug, space) 常驻。
#
# 为什么缓存字符串而不是缓存 dict：序列化是这条路上最贵的一段。实测一条
# 4842 帧的素材（4.75 MB）：describe 590ms + json.dumps 597ms，而 Flask 的
# jsonify 拿到 dict 之后**还会再序列化一遍**（又 ≈600ms）——加起来 1.8 秒，
# 也就是"切一条素材要等近两秒"的全部来源。返回字符串让 Flask 直接吐出去，
# 既省掉重复序列化，又让**切回已经看过的素材变成零成本**。
_JSON_CACHE: dict[tuple[str, str, int | None, bool], str] = {}


def describe_json(slug: str, space: str = "global", tid: int | None = None,
                  lite: bool = False) -> str | None:
    """describe() 的 JSON 文本。读不出来返回 None（调用方照旧走 404）。"""
    key = (slug, "incam" if space == "incam" else "global", tid, lite)
    with _LOCK:
        hit = _JSON_CACHE.get(key)
        if hit is not None:
            return hit
    d = describe(slug, key[1], tid, lite=lite)
    if not d.get("ok"):
        return None
    # separators 去掉分隔符后的空格：一帧 123 个数，省下的大约就是 600 KB
    text = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    with _LOCK:
        _JSON_CACHE[key] = text
    return text


def clear(slug: str | None = None) -> None:
    with _LOCK:
        keys = list(_CACHE) if slug is None else [k for k in _CACHE if k[0] == slug]
        for k in keys:
            del _CACHE[k]
        jkeys = list(_JSON_CACHE) if slug is None else [k for k in _JSON_CACHE if k[0] == slug]
        for k in jkeys:
            del _JSON_CACHE[k]
