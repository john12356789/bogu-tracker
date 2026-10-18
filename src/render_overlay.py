"""把重建的 mesh 渲染并叠加回原视频，输出 mp4。

为什么自己写渲染器
------------------
GVHMR 出图用的是 pytorch3d 的 CUDA 光栅化器，而 Meta 没发 Windows 版
（这正是我们写 shim 的原因）。但那个渲染器只做了一件事：把三角形画到图上。
这里用画家算法自己做——按深度排序后逐三角形填充，纯 OpenCV，零 GPU 依赖。

视觉约定
--------
**骨盆区域被染成热色（红），身体其余部分是冷灰蓝。** 这是刻意的：
你要看的不是"动作流不流畅"（那个一定流畅），而是"胯部那块 mesh 到底贴不贴人"。
把最不可信的区域标出来，眼睛才有地方盯。

用法
----
    python render_overlay.py GVHMR/outputs/demo/dance_1280p/hmr4d_results_incam.npz \
        --video processed/dance_1280p.mp4 -o overlay.mp4

    # 快速预览前 60 帧
    python render_overlay.py ... -o preview.mp4 --max-frames 60

    # 半分辨率，快一倍
    python render_overlay.py ... -o overlay.mp4 --scale 0.5

自己拼格子（--views）
---------------------
`--layout` 那五个是固定组合，够用但只能整组选。要看"到底哪一段预测得不准"，
得能自己挑格子、自己挑时间段：

    # 只要叠加 + 侧面，两格一行
    --views overlay,world@90

    # 2×2 网格：原片 | 叠加 | 正面 | 侧后 45°
    --views raw,overlay,world@0,world@45 --grid 2x2

    # 只看第 12.4~15.9 秒（胯部遮挡那一段）
    --range 12.4:15.9

视图 id：`raw`（原片）、`overlay`（mesh 投在原片上，相机系）、
`world[@角度]`（世界系 mesh + 地面网格；0=正面、90=侧面，和 --layout 的
正面/侧面完全一致）。同一个 id 可以出现多次，顺序就是排布顺序。

**`--layout` 就是这两三个视图的简写**，展开关系见 LAYOUT_VIEWS：
    overlay     → overlay
    sidebyside  → raw,overlay
    triple      → raw,overlay,world@0
    quad        → raw,overlay,world@0,world@90
所以旧命令的输出逐字节不变（`--views` 那条路和 `--layout` 那条路调的是
同一套格子代码，不是两份实现）。

时间段（--range）
-----------------
`--range 12.4:15.9`（秒，可省略一端）或 `--frames 372:477`（帧号，精确）。
`--max-frames N` 是"再截前 N 帧"，和上面两个可以叠加。

为什么需要它：**整段渲染贵，而且判断精度时你要盯的就是那几秒**。一条 4 分半的
素材 4 格渲染约 35 分钟（实测 0.2~0.3 秒/帧），只渲 3 秒就几秒钟的事。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_npz import mesh_path_for  # noqa: E402

# 骨盆热区半径（米）：离骨盆区域关节多远以内算作"胯部"
PELVIS_RADIUS = 0.22
COLD = np.array([150, 165, 180], dtype=np.float64)   # 冷：身体其余部分（BGR）
HOT = np.array([70, 70, 255], dtype=np.float64)      # 热：骨盆区域（BGR）

# 两组结果的配色（BGR）：A 偏蓝，B 偏橙。冷=身体其余，热=骨盆区域。
COLOR_A = {"cold": np.array([180, 140, 80], dtype=np.float64),
           "hot": np.array([90, 70, 245], dtype=np.float64), "label": "ViTPose"}
COLOR_B = {"cold": np.array([80, 175, 215], dtype=np.float64),
           "hot": np.array([45, 195, 255], dtype=np.float64), "label": "RTMPose"}



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把重建 mesh 叠加回原视频")
    p.add_argument("npz", help="来自 export_gvhmr.py 的 incam npz")
    p.add_argument("--video", required=True, help="对应的原始视频")
    p.add_argument("--global-npz", default=None,
                   help="世界系 npz。给了它才能出「纯 mesh + 地面网格」这一格")
    p.add_argument("-o", "--out", default="overlay.mp4")
    p.add_argument("--compare-npz", default=None,
                   help="第二组结果的 incam npz。给了它就进入叠加对比模式")
    p.add_argument("--compare-global-npz", default=None,
                   help="第二组结果的 global npz（对比模式必需）")
    p.add_argument("--layout", choices=["overlay", "sidebyside", "triple", "quad", "compare"],
                   default=None,
                   help="compare=四格对比：原片 | 两组叠加 | 正面 | 侧面")
    p.add_argument("--views", default=None,
                   help="自己拼格子：逗号分隔的视图 id，如 overlay,world@90,raw。"
                        "world 可以带朝向（度，0=正面 90=侧面）。给了它就别再给 --layout")
    p.add_argument("--grid", default=None, metavar="RxC",
                   help="网格排布，如 2x2。不给就全排成一行")
    p.add_argument("--cell-h", type=int, default=None, metavar="PX",
                   help="每格统一缩到的高度（默认=原片高）。格子多的时候调小它，"
                        "否则 4 格 720p 拼出来就是 2880 宽，浏览器里根本看不清")
    p.add_argument("--range", default=None, metavar="T0:T1",
                   help="只渲染这一段（秒），如 12.4:15.9；一端可省")
    p.add_argument("--frames", default=None, metavar="I0:I1",
                   help="同上，但用帧号（精确，不受 --fps 取整影响）")
    p.add_argument("--alpha", type=float, default=0.55, help="叠加格 mesh 不透明度")
    p.add_argument("--scale", type=float, default=1.0, help="渲染缩放（默认 1.0）")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--audio", default=None,
                   help="从该文件复制音轨到输出。渲染是裸帧管道，本身不携带音频，"
                        "要给成品配音就得在这里显式指定源（通常给原片）")
    p.add_argument("--lit-only", action="store_true", help="只画受光面，背面剔除更激进")
    p.add_argument("--grid-step", type=float, default=0.5, help="地面网格间距（米）")
    p.add_argument("--grid-extent", type=float, default=4.0, help="地面网格半径（米）")
    p.add_argument("--workers", type=int, default=0,
                   help="逐帧渲染的并行进程数。0=自动（留 2 个核给 x264），1=单进程。"
                        "渲染是逐帧独立的，多进程能把单线程的三角形填充摊到多个核上；"
                        "输出与单进程逐字节相同（帧序由主进程串行写进 ffmpeg）")
    return p.parse_args()


class FrameSource:
    """顺序取帧，一次只在内存里留一帧。

    **以前是把整段帧读进 list**。720×1280×3 = 2.76 MB/帧 = 83 MB/秒：
    BV1CC4d6vE4m（268s / 8050 帧）光这一项就是 22 GB，比这台机器的物理内存
    （15.9 GB）还大——那条 quad 是靠页面文件硬撑出来的。渲染每帧要 0.2~0.3 秒，
    解码每帧才几毫秒，**边渲染边解码永远不会成为瓶颈**，没有任何理由为了
    "以后可能重看某一帧"把整段扣在内存里。

    调用契约：按**递增帧号**顺序 read()（render_parallel 的分发循环就是递增的，
    见那里的说明）。它不提供随机访问，所以这里是顺序解码而不是 seek。
    """

    def __init__(self, path: str, start: int = 0, limit: int | None = None,
                 scale: float = 1.0):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"打不开视频：{path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.scale = scale
        self.left = limit
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def read(self) -> np.ndarray | None:
        """下一帧；到尾部（或读满 limit）返回 None。"""
        if self.left is not None and self.left <= 0:
            return None
        ok, f = self.cap.read()
        if not ok:
            return None
        if self.left is not None:
            self.left -= 1
        if self.scale != 1.0:
            f = cv2.resize(f, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA)
        return f

    def close(self) -> None:
        self.cap.release()


def resolve_range(args) -> tuple[int, int]:
    """要渲染哪一段，返回 (起始帧, 结束帧)，结束帧 -1 表示到结尾。

    `--frames` 优先于 `--range`：帧号是精确的，秒要过一道 fps 取整（`I33` 那种
    取整误差就是项目里踩过的坑，能不用秒就不用秒）。帧数必须等知道 npz 有多少帧
    才能定，所以这里只返回端点——收口在 clamp_range。
    """
    if args.frames:
        spec, per_unit = args.frames, 1.0
    elif args.range:
        spec, per_unit = args.range, args.fps
    else:
        return 0, -1

    parts = spec.split(":")
    if len(parts) != 2:
        raise SystemExit(f"{spec!r} 应该写成 T0:T1 或 I0:I1（其中一端可以省）")

    def conv(raw: str, default: int) -> int:
        raw = raw.strip()
        return default if not raw else int(round(float(raw) * per_unit))

    return conv(parts[0], 0), conv(parts[1], -1)


def clamp_range(start: int, end: int, n_npz: int, max_frames: int | None) -> tuple[int, int]:
    """把 (起始, 结束) 收进 [0, n_npz]，返回 (起始帧, 帧数)。"""
    if end < 0:
        end = n_npz
    start = max(0, min(start, n_npz))
    end = max(start, min(end, n_npz))
    count = end - start
    if max_frames is not None:
        count = min(count, max_frames)
    return start, count


# ---------------------------------------------------------------------------
# 格子：视图 id → 一格画面；多格 → 一张画布
# ---------------------------------------------------------------------------
# 一个 id 一件事。以前这些都混在 make_frame_builder 的
# `if panels >= 2: cols.insert(0, frame)` 里——那是"五个固定组合"，不是视图。
CELL_KINDS = ("raw", "overlay", "world")

# --layout 的五个固定组合，展开成视图列表。**legacy 那条路走的就是这套格子代码**，
# 不是另有一份实现——否则两条路的输出迟早会不一样。
LAYOUT_VIEWS: dict[str, list[str]] = {
    "overlay": ["overlay"],
    "sidebyside": ["raw", "overlay"],
    "triple": ["raw", "overlay", "world@0"],
    "quad": ["raw", "overlay", "world@0", "world@90"],
}


def parse_views(spec: str) -> list[dict]:
    """`raw,overlay,world@90` → [{kind,yaw}...]。顺序即排布顺序。"""
    cells: list[dict] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        kind, _, yaw = tok.partition("@")
        kind = kind.strip().lower()
        if kind not in CELL_KINDS:
            raise SystemExit(f"未知视图 {kind!r}，可选：{', '.join(CELL_KINDS)}"
                             f"（world 可以带朝向，如 world@90）")
        cell = {"kind": kind, "yaw": 0.0}
        if yaw.strip():
            try:
                cell["yaw"] = float(yaw)
            except ValueError:
                raise SystemExit(f"{tok!r} 的朝向不是数字：{yaw!r}（度，0=正面 90=侧面）")
        cells.append(cell)
    if not cells:
        raise SystemExit("--views 是空的")
    return cells


def parse_grid(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    a, _, b = spec.lower().partition("x")
    try:
        r, c = int(a), int(b)
    except ValueError:
        raise SystemExit(f"--grid 应该写成 RxC，如 2x2（给的是 {spec!r}）")
    if r < 1 or c < 1:
        raise SystemExit(f"--grid 不能有 0：{spec!r}")
    return r, c


def yaw_dir(yaw_deg: float, facing: np.ndarray) -> np.ndarray:
    """把"正对相机的方向"绕 Y 轴转 yaw 度。

    0 就是正对相机（旧 `--layout` 的正面），**90 等于旧的侧面** —— 旧的
    `cam_side = cross(up, facing)` 正好是 facing 绕 Y 转 +90°，所以两套编号能对上，
    对照旧输出时不用换算。
    """
    t = np.radians(yaw_deg)
    c, s = float(np.cos(t)), float(np.sin(t))
    v = np.array([c * facing[0] + s * facing[2], facing[1],
                  -s * facing[0] + c * facing[2]])
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else facing


def _pad_even(canvas: np.ndarray) -> np.ndarray:
    """把画布补到**宽高都是偶数**（右下各补一列/一行黑）。

    这不是美学问题：**libx264 配 `-pix_fmt yuv420p`（4:2:0 色度二次采样）要求
    宽高都是偶数**，奇数会直接 EINVAL——一个包都不写，整段渲染白跑，日志里只有
    一句 ffmpeg 的报错。而画布尺寸是"每格等比缩放后的宽度之和 / 槽宽 × 列数"，
    随手就是奇数（实测：预览渲染崩在这里）。

    补在**总的画布**上而不是每格上：每格各补一条会在格子之间留出黑缝；补总画布
    只在最右/最下多一列像素，看不出来。而且尺寸对每一帧都是同一个值（每格的大小
    只由 cell_h 决定），所以并行那条路（worker 按首帧的 shape 开共享内存）不会
    因为某一帧补了、另一帧没补而错位。
    """
    h, w = canvas.shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return canvas
    return np.pad(canvas, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=0)


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    """按高度等比缩放。一行的格子要等高才能 hstack。"""
    ih, iw = img.shape[:2]
    if ih == h:
        return img
    return cv2.resize(img, (max(1, round(iw * h / ih)), h), interpolation=cv2.INTER_AREA)


def _fit_slot(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """等比缩放到能放进 h×w，居中，空隙留黑。

    **不直接拉伸。** 拿来看"mesh 贴不贴人"的图被压扁过之后，判断就不作数了——
    宁可留黑边，也不要一种看不出来的失真。
    """
    ih, iw = img.shape[:2]
    s = min(h / ih, w / iw)
    r = cv2.resize(img, (max(1, round(iw * s)), max(1, round(ih * s))),
                   interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def assemble(cells: list[np.ndarray], grid: tuple[int, int] | None, cell_h: int) -> np.ndarray:
    """把若干格拼成一张画布。

    不给 --grid 就是一行（等价于旧的 hstack，每格用自己的宽高比）；
    给了就按 R×C 排，槽宽取最宽那格缩放后的宽度，各格等比放进去。

    **槽位富余时裁掉**：要 1×4 却只有 2 格，就该出 1×2，右边不该留两条黑边——
    那些黑格没有任何信息，只是把画布撑宽、把每一格都挤小。裁的是"没用到的
    整行/整列"，2×2 只放 3 格时右下那格仍然留黑（网格形状是明确要求的，
    缺的那格本身就说明少了一个视图）。
    """
    if grid is None:
        return _pad_even(np.hstack([_resize_h(c, cell_h) for c in cells]))
    rows, cols = grid
    if len(cells) > rows * cols:
        print(f"⚠ {len(cells)} 个视图放不进 {rows}x{cols} 网格，多出来的丢弃",
              file=sys.stderr)
        cells = cells[:rows * cols]
    used_r = (len(cells) - 1) // cols + 1
    # 列数取"最满那一行"的格数：1×4 放 2 格 → 1×2；2×2 放 3 格 → 仍是 2×2
    used_c = max(min(cols, len(cells) - r * cols) for r in range(used_r))
    slot_w = max(_resize_h(c, cell_h).shape[1] for c in cells)
    canvas = np.zeros((used_r * cell_h, used_c * slot_w, 3), np.uint8)
    for k, c in enumerate(cells):
        r, col = divmod(k, cols)
        canvas[r * cell_h:(r + 1) * cell_h, col * slot_w:(col + 1) * slot_w] = \
            _fit_slot(c, cell_h, slot_w)
    return _pad_even(canvas)


def build_decimator(verts_ref: np.ndarray, faces: np.ndarray, target_faces: int = 4000):
    """顶点聚类降采样。返回 (D, new_faces)。

    SMPL-X 有 20908 个三角形。在纯 Python 驱动的画家算法下，每帧要为两万个多边形
    各调一次 fillPoly——实测占渲染耗时的 95%。降到几千面对本项目输出分辨率而言
    视觉上几乎无差别，但速度提升数倍。

    聚类关系只取决于网格拓扑，与姿态无关，所以只算一次；逐帧用稀疏矩阵 D 把
    (10475,3) 的顶点聚成 (n_new,3)，等价于对每个簇取成员均值。
    """
    from scipy import sparse

    lo = verts_ref.min(axis=0)
    size = verts_ref.max(axis=0) - lo
    cell = float(size.max()) / 60.0
    inv = nf = None
    for _ in range(50):
        g = np.floor((verts_ref - lo) / cell).astype(np.int64)
        ids = g[:, 0] + g[:, 1] * 1000 + g[:, 2] * 1_000_000
        uniq, inv = np.unique(ids, return_inverse=True)
        cand = np.unique(np.sort(inv[faces], axis=1), axis=0)
        nf = cand[(cand[:, 0] != cand[:, 1]) & (cand[:, 1] != cand[:, 2])
                  & (cand[:, 0] != cand[:, 2])]
        if len(nf) <= target_faces:
            break
        cell *= 1.12

    n_new = int(inv.max()) + 1
    D = sparse.csr_matrix(
        (np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(n_new, len(inv))
    )
    counts = np.asarray(D.sum(axis=1)).ravel()
    D = sparse.diags(1.0 / np.maximum(counts, 1.0)) @ D
    return D.tocsr().astype(np.float32), np.ascontiguousarray(nf, dtype=np.uint32)


def vertex_heat(verts: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """每个顶点到骨盆区域的距离 -> 0..1 的热度（1 = 最热）。

    骨盆区域关节取双胯 + 盆骨（SMPL 索引 0/1/2）。用距离而不是部位分割，
    是因为这不需要额外的分割文件，且过渡更自然。
    """
    pelvis_pts = joints[[0, 1, 2]]                      # (3, 3)
    d = np.linalg.norm(verts[:, None, :] - pelvis_pts[None, :, :], axis=2).min(axis=1)
    return np.clip(1.0 - d / PELVIS_RADIUS, 0.0, 1.0)




def render_meshes(
    bg: np.ndarray,
    specs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    K: np.ndarray,
    alpha: float,
    lit_only: bool = False,
) -> np.ndarray:
    """把多个 mesh 一起画到 bg 上。

    specs 每项为 (verts, faces, heat, cold_bgr, hot_bgr)，顶点需已在目标坐标系。

    **关键**：两个 mesh 的三角形必须一起按深度排序后统一绘制，否则它们之间的
    遮挡关系会取决于绘制顺序——远处的 mesh 可能盖住近处的。逐 mesh 分开画是错的。
    """
    h, w = bg.shape[:2]
    tris, p2d, depths, cols = [], [], [], []

    for verts, faces, heat, cold, hot in specs:
        z = verts[:, 2]
        proj = verts @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = proj[:, :2] / proj[:, 2:3]
        zt = z[faces]
        keep = (zt > 0.05).all(axis=1)
        f = faces[keep]
        if len(f) == 0:
            continue
        tris.append(f)
        p2d.append(uv[f].astype(np.int32))
        depths.append(zt[keep].mean(axis=1))

        v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(nn < 1e-9, 1.0, nn)
        light = np.array([0.35, -0.5, -0.8])
        light /= np.linalg.norm(light)
        shade = (0.35 + 0.65 * np.abs(n @ light))[:, None]

        fh = heat[f].mean(axis=1)[:, None]
        base = cold[None, :] * (1 - fh) + hot[None, :] * fh
        cols.append(np.clip(base * shade, 0, 255).astype(np.uint8))

    if not tris:
        return bg

    pts = np.concatenate(p2d)                  # (F, 3, 2) int32
    depth = np.concatenate(depths)
    col = np.concatenate(cols)
    order = np.argsort(-depth)                 # 远 -> 近，画家算法
    p_o, c_o = pts[order], col[order]

    # ---- 单趟绘制：蒙版直接来自 BGRA 的 alpha 通道 ----
    # 分成「图层」和「蒙版」两趟会各要一次全量 fillPoly，实测两趟合计占 95% 耗时。
    # 改成往 BGRA 图上画、每次带上 alpha=255，画完的 alpha 通道**就是**蒙版——
    # 一趟搞定，且蒙版天然只覆盖被画到的像素。
    layer = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    key = c_o.astype(np.uint32) >> 4
    k = (key[:, 0] << 16) | (key[:, 1] << 8) | key[:, 2]
    cuts = np.flatnonzero(np.diff(k) != 0) + 1
    bounds = np.concatenate(([0], cuts, [len(k)]))
    for s, e in zip(bounds[:-1], bounds[1:]):
        # 必须传 numpy 数组的列表；.tolist() 会变成嵌套 Python 列表，cv2 解析不了
        polys = list(p_o[s:e]) if e - s > 1 else [p_o[s]]
        b, g_, r = c_o[s].tolist()
        cv2.fillPoly(layer, polys, (b, g_, r, 255))

    # ---- 混合 ----
    # 关键：**不要用 NumPy 的布尔索引**（layer[mask]）——它在这个尺寸上要 7.7ms，
    # 比全帧做一遍还慢。改用 OpenCV 的 C 实现：addWeighted 全帧算混合，
    # copyTo 按掩码把结果贴回去，合计约 1.1ms。
    mask = layer[:, :, 3]
    out = bg.copy()
    if alpha >= 0.999:
        cv2.copyTo(layer[:, :, :3], mask, out)
    else:
        blend = cv2.addWeighted(layer[:, :, :3], alpha, bg, 1.0 - alpha, 0.0)
        cv2.copyTo(blend, mask, out)
    return out


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """构造 world->camera 的 4x4（OpenCV 约定：+X 右，+Y 下，+Z 前）。"""
    eye, target, up = map(lambda a: np.asarray(a, dtype=np.float64), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    E = np.eye(4)
    E[:3, 0], E[:3, 1], E[:3, 2], E[:3, 3] = r, d, f, eye
    return np.linalg.inv(E)          # 世界->相机


def draw_ground_grid(img, K, w2c, ground_y, cx, cz, step, extent, color=(110, 110, 120)):
    """在世界系 y=ground_y 平面上画网格，按相机投影后画线。"""
    h, w = img.shape[:2]

    def project(pts):
        cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
        if (cam[:, 2] <= 0.05).any():
            return None
        p = (K @ cam.T).T
        uv = p[:, :2] / p[:, 2:3]
        return uv

    n = int(extent / step)
    for i in range(-n, n + 1):
        off = i * step
        segments = (
            np.array([[cx + off, ground_y, cz - extent], [cx + off, ground_y, cz + extent]]),
            np.array([[cx - extent, ground_y, cz + off], [cx + extent, ground_y, cz + off]]),
        )
        for seg in segments:
            uv = project(seg)
            if uv is None:
                continue
            p0 = tuple(np.round(uv[0]).astype(int))
            p1 = tuple(np.round(uv[1]).astype(int))
            # 极端投影（点几乎在相机平面上）会甩到很远处，直接跳过而不是画一条长线
            if max(abs(p0[0]), abs(p1[0])) > 20 * w or max(abs(p0[1]), abs(p1[1])) > 20 * h:
                continue
            cv2.line(img, p0, p1, color, 1 if i % 2 else 2, cv2.LINE_AA)
    return img


def body_facing(joints: np.ndarray) -> np.ndarray:
    """从双胯的相对位置估出身体朝向（水平面内的单位向量）。

    取 right_hip - left_hip 得到"人的右侧"方向，再与竖直方向叉乘得到前方。
    右手系下若面朝 +Z、上方 +Y，则右侧为 -X；故 forward = up × right。
    """
    right_vec = joints[2] - joints[1]           # 右胯 - 左胯
    right_vec[1] = 0.0                           # 只取水平分量
    n = np.linalg.norm(right_vec)
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    right_vec /= n
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), right_vec)
    n = np.linalg.norm(fwd)
    return fwd / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def build_world_panel(
    meshes: list[tuple[np.ndarray, np.ndarray, dict]],
    faces: np.ndarray,
    size: tuple[int, int],
    grid_center: tuple[float, float],
    grid_step: float,
    grid_extent: float,
    facing: np.ndarray | None = None,
    distance: float = 3.2,
    eye_height: float = 1.5,
) -> np.ndarray:
    """世界系：若干 mesh + 地面网格，背景纯净，不受原视频干扰。

    meshes 每项为 (verts, joints, color_dict)。
    facing 给定时相机放在身体正前方——GVHMR 世界系的偏航是对齐第一帧的，
    舞者朝哪边本来就任意，用固定偏移会侧面看人。
    """
    w, h = size
    cx, cz = grid_center

    if facing is None:
        facing = np.array([0.0, 0.0, 1.0])
    eye = np.array([cx, eye_height, cz]) + facing * distance
    target = np.array([cx, 0.95, cz])
    w2c = look_at(eye, target)

    focal = 1.6 * w                      # 约 40° 视场
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)

    bg = np.full((h, w, 3), 42, dtype=np.uint8)
    draw_ground_grid(bg, K, w2c, 0.0, cx, cz, grid_step, grid_extent)

    # 顶点转到相机系后交给多 mesh 渲染器统一排序绘制
    specs = []
    for verts, joints, col in meshes:
        cam_v = verts @ w2c[:3, :3].T + w2c[:3, 3]
        specs.append((cam_v, faces, vertex_heat(verts, joints), col["cold"], col["hot"]))
    return render_meshes(bg, specs, K, alpha=1.0)


def render_cell(cell: dict, per: dict, ctx: dict) -> np.ndarray:
    """一格画面。kind 只决定"画什么"，不决定"排在哪"——排布归 assemble。"""
    kind = cell["kind"]
    frame = per["frame"]
    if kind == "raw":
        return frame
    if kind == "overlay":
        return render_meshes(
            frame,
            [(per["verts"], ctx["faces"],
              vertex_heat(per["verts"], per["joints"]), COLD, HOT)],
            per["K"], ctx["alpha"])
    if kind == "world":
        if per["gv"] is None:
            raise RuntimeError("world 视图需要 --global-npz")
        return build_world_panel(
            [(per["gv"], per["gj"], {"cold": COLD, "hot": HOT})],
            ctx["faces"], (ctx["w"], ctx["h"]),
            ctx["grid_center"], ctx["grid_step"], ctx["grid_extent"],
            facing=yaw_dir(cell["yaw"], ctx["cam_facing"]))
    raise RuntimeError(f"视图 {kind} 没有实现")


def render_compare_cell(per: dict, ctx: dict) -> np.ndarray:
    """A/B 对比的一行：原片 | 两组叠加 | 正面 | 侧面。

    还是旧的四格，只是从布局分支里挪出来变成"一种格子"。
    """
    faces, alpha = ctx["faces"], ctx["alpha"]
    w, h = ctx["w"], ctx["h"]
    specs_v = [(per["verts"], faces, vertex_heat(per["verts"], per["joints"]),
                COLOR_A["cold"], COLOR_A["hot"])]
    if per["b_verts"] is not None:
        specs_v.append((per["b_verts"], faces,
                        vertex_heat(per["b_verts"], per["b_joints"]),
                        COLOR_B["cold"], COLOR_B["hot"]))
    cols = [per["frame"], render_meshes(per["frame"], specs_v, per["K"], alpha)]
    for fdir in (ctx["cam_facing"], ctx["cam_side"]):
        ms = []
        if per["gv"] is not None:
            ms.append((per["gv"], per["gj"], COLOR_A))
        if per["b_gv"] is not None:
            ms.append((per["b_gv"], per["b_gj"], COLOR_B))
        cols.append(build_world_panel(ms, faces, (w, h), ctx["grid_center"],
                                      ctx["grid_step"], ctx["grid_extent"],
                                      facing=fdir))
    return np.hstack(cols)


def make_frame_builder(ctx: dict):
    """把「逐帧组装画面」的逻辑封成一个只依赖当帧数据的纯函数。

    多进程要靠它：ctx 里全是**与帧无关**的常量（格子表、网格、相机朝向、配色、
    尺寸），每帧只需带上当帧的 frame / verts / joints / K。这样任务体很小，
    不必把整段序列的顶点数组复制给每个 worker。

    输出必须与单进程路径逐字节一致——两条路径调的是同一个 build()。
    """
    if ctx.get("compare"):
        build_cells = lambda per: render_compare_cell(per, ctx)   # noqa: E731
    else:
        cells, grid, cell_h = ctx["cells"], ctx["grid"], ctx["cell_h"]
        build_cells = lambda per: assemble(                            # noqa: E731
            [render_cell(c, per, ctx) for c in cells], grid, cell_h)

    return build_cells


def _worker_entry(ctx, slot_names, slot_shape, in_q, done_q):
    """worker 主循环：从队列取帧号，渲染进指定共享槽，回报完成。

    渲染结果写进**共享内存槽**而不是通过队列回传——一帧 quad 是 11MB，
    1119 帧走 pickle 回传要复制十几 GB，会把并行省下的时间又吃回去。

    必须留住 SharedMemory 对象本身：写成 `np.ndarray(buffer=SharedMemory(name=n).buf)`
    的话，那个临时对象在表达式结束后立刻被 GC，memoryview 随之失效，
    下一行就会抛 `ValueError: operation forbidden on released memoryview object`。
    """
    # OpenCV 默认按核数开内部线程。多个 worker 叠加就变成「worker 数 × 核数」
    # 个线程抢同一批核（8×12=96 个线程抢 12 核），并行效率被上下文切换吃掉。
    # 每个 worker 限成单线程，让并行度只由 worker 数决定。
    cv2.setNumThreads(1)
    build = make_frame_builder(ctx)
    shms = [shared_memory.SharedMemory(name=nm) for nm in slot_names]
    slots = [np.ndarray(slot_shape, dtype=np.uint8, buffer=s.buf) for s in shms]
    try:
        while True:
            task = in_q.get()
            if task is None:
                return
            i, slot_idx, per = task
            slots[slot_idx][:] = build(per)
            done_q.put(i)
    finally:
        for s in shms:
            s.close()


def render_parallel(n, pack, ctx, slot_shape, workers, sink, out_w, h):
    """逐帧并行渲染，帧序由本进程串行写进 ffmpeg —— 所以输出与单进程完全一致。

    用「固定数量共享槽 + 有界队列」做背压：在途帧数不超过槽数，
    完成的帧若还没轮到它写出，就占着它的槽等着——这样内存有上界（槽数×帧大小），
    不会因为某几帧算得快就把结果堆爆内存。

    `pack(i)` 按**递增 i** 调用（就是下面那个分发循环），返回 None 表示没有第 i 帧
    了（视频比 npz 短），此时就地收尾、输出短一点，而不是拿旧帧凑数。
    """
    size = int(np.prod(slot_shape))
    n_slots = max(workers + 2, min(2 * workers + 4, n))
    shms = [shared_memory.SharedMemory(create=True, size=size) for _ in range(n_slots)]
    names = [s.name for s in shms]

    in_q = mp.Queue(maxsize=n_slots + workers + 4)
    done_q = mp.Queue()
    procs = [mp.Process(target=_worker_entry,
                        args=(ctx, names, slot_shape, in_q, done_q), daemon=True)
             for _ in range(workers)]
    for p in procs:
        p.start()

    free = list(range(n_slots))
    assigned: dict[int, int] = {}
    completed: set[int] = set()
    nxt = 0
    last_recv = time.time()

    def wait_done():
        """取一个完成信号。**绝不无限等**——卡死比报错难查得多（实测丢过 5 小时）。"""
        nonlocal last_recv
        while True:
            try:
                v = done_q.get(timeout=1.0)
                last_recv = time.time()
                return v
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    raise RuntimeError("渲染 worker 全部退出但帧未收齐——worker 崩溃了")
                if time.time() - last_recv > 120:
                    raise RuntimeError(
                        "已 120 秒收不到任何渲染结果，判定 worker 起不来。"
                        "Windows 用 spawn 起子进程要重新导入 __main__，所以本脚本"
                        "**不能通过管道喂给 python**（python - <<EOF 或 python -c），"
                        "必须作为真实文件运行：python src/render_overlay.py ...")

    def flush():
        nonlocal nxt
        while nxt in completed:
            s = assigned.pop(nxt)
            sink(shms[s].buf)
            free.append(s)
            completed.discard(nxt)
            nxt += 1
            if nxt % 25 == 0 or nxt == n:
                print(f"  渲染 {nxt}/{n}", end="\r", flush=True)

    try:
        for i in range(n):
            while not free:
                completed.add(wait_done())
                flush()
            per = pack(i)
            if per is None:
                n = i                     # 视频先到头：后面没有了，别再发任务
                break
            s = free.pop()
            assigned[i] = s
            in_q.put((i, s, per))
        while nxt < n:
            completed.add(wait_done())
            flush()
    finally:
        for _ in procs:
            try:
                in_q.put_nowait(None)
            except Exception:
                pass
        for p in procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        for s in shms:
            s.close()
            try:
                s.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"找不到 {npz_path}", file=sys.stderr)
        return 1

    def load_set(p, want_global: bool):
        # 顶点在**另一个文件**里（`x_incam.npz` -> `x_incam.mesh.npz`，见
        # mesh_npz.py）：它一个就占原 npz 的 99.5%，而只有渲染读它，所以平时
        # 不落盘。这里两边都开。
        #
        # **mmap**。顶点是这里最大的一块：BV1CC4d6vE4m 的 incam 顶点
        # 8050×10475×3 = 0.94 GB，加 global 那份接近 2 GB。
        #
        # 说清楚它**不是**在降峰值：整段顶点都会被顺序摸一遍，内存宽裕时系统
        # 会把页都留在工作集里，实测峰值 2.1 GB 一点没变（我量过）。
        # 它买到的是**可回收**：这些是文件映射的干净页，机器紧张时系统直接丢掉，
        # 不需要换出到页面文件。而以前那份帧列表是匿名内存，紧张时只能交换——
        # 那条 268 秒的素材全片渲染要 22 GB，就是靠页面文件硬撑的。
        d = np.load(p, mmap_mode="r")
        m = np.load(mesh_path_for(p), mmap_mode="r")
        out = {"verts": m["vertices"], "joints": d["joints"],
               "faces": np.asarray(m["faces"])}
        K = d["K"]
        out["K"] = np.tile(K, (len(m["vertices"]), 1, 1)) if K.ndim == 2 else K
        if want_global and "cam_to_world" in d:
            out["c2w"] = d["cam_to_world"]
        return out

    def load_world(p):
        """世界系那一份：只要 vertices + joints，同样分在两个文件里。"""
        return (np.load(mesh_path_for(p), mmap_mode="r")["vertices"],
                np.load(p, mmap_mode="r")["joints"])

    compare = bool(args.compare_npz)
    if compare and not args.compare_global_npz:
        print("对比模式需要 --compare-global-npz。", file=sys.stderr)
        return 1

    # ---- 视图：--views 自己拼；--layout 是几个固定组合的简写 ----
    #
    # **要在加载之前算出来**：只有"要画 mesh 的格子"（overlay/world）才需要网格
    # 文件，纯 raw 的格子不用，那样就不该拦着人家。
    if args.views and args.layout:
        print("--views 和 --layout 只能给一个（--layout 就是视图列表的简写）",
              file=sys.stderr)
        return 2
    layout = args.layout or ("compare" if compare else ("quad" if not args.views else None))
    if compare and (args.views or args.grid):
        # 对比模式那四格是固定的（两组重建各画一遍），--views/--grid 落不进去。
        # 与其默默忽略，不如说清楚。
        print("对比模式（--compare-npz）不支持 --views/--grid：它的四格是固定的",
              file=sys.stderr)
        return 2
    if args.views:
        cells = parse_views(args.views)
    elif layout == "compare":
        cells = []                       # 对比模式走自己那条格子路径
    else:
        cells = parse_views(",".join(LAYOUT_VIEWS[layout]))

    # ---- 网格文件在不在 ----
    if compare or any(c["kind"] != "raw" for c in cells):
        want = [npz_path]
        for extra in (args.compare_npz, args.global_npz, args.compare_global_npz):
            if extra:
                want.append(Path(extra))
        miss = [mesh_path_for(p) for p in want if not mesh_path_for(p).exists()]
        if miss:
            print(f"网格文件不在：{miss[0]}\n"
                  "  顶点占了结果文件的 99.5%，而只有渲染读它，所以平时不落盘"
                  "（见 src/mesh_npz.py）。\n"
                  "  补一份：python src/export_gvhmr.py <那个 hmr4d_results.pt> "
                  "--spaces incam[,global]    （约 20~60 秒）\n"
                  "  走界面渲染的话这一步会自动做，看到这条通常是在手敲命令行。",
                  file=sys.stderr)
            return 1

    A = load_set(npz_path, False)
    faces = A["faces"]
    print(f"A 组: {A['verts'].shape[1]} 顶点 / {len(faces)} 面，{len(A['verts'])} 帧")

    B = load_set(Path(args.compare_npz), False) if compare else None
    gA = gB = None
    if args.global_npz:
        gA = load_world(Path(args.global_npz))
        print(f"A 组世界系: {len(gA[0])} 帧")
    if compare and args.compare_global_npz:
        gB = load_world(Path(args.compare_global_npz))
        print(f"B 组世界系: {len(gB[0])} 帧")
    elif any(c["kind"] == "world" for c in cells) and gA is None:
        # --layout 是简写，缺了降级（旧行为）；--views 是明确点名要的，直接报错——
        # 悄悄把用户点的那格换成别的东西，比报错难查得多。
        if args.views:
            print("world 视图需要 --global-npz（世界系 npz）", file=sys.stderr)
            return 1
        print(f"⚠ {layout} 需要 --global-npz，已降级为 sidebyside", file=sys.stderr)
        layout = "sidebyside"
        cells = parse_views(",".join(LAYOUT_VIEWS[layout]))

    # ---- 时间段：所有数组切同一段，视频也只解码这一段 ----
    n_max = len(A["verts"])
    for extra in (B["verts"] if B is not None else None,
                  gA[0] if gA is not None else None,
                  gB[0] if gB is not None else None):
        if extra is not None:
            n_max = min(n_max, len(extra))
    start, count = clamp_range(*resolve_range(args), n_max, args.max_frames)
    if count <= 0:
        print("这段时间里没有帧（检查 --range/--frames 是否超出素材长度）", file=sys.stderr)
        return 1
    seg = slice(start, start + count)

    src = FrameSource(args.video, start, count, args.scale)
    fps = src.fps
    first = src.read()                    # 顺便当尺寸探针，见下面的 first_canvas
    if first is None:
        print(f"视频里读不到第 {start} 帧：{args.video}", file=sys.stderr)
        return 1
    h, w = first.shape[:2]
    print(f"视频: 第 {start}~{start + count - 1} 帧（共 {count} 帧）"
          f" @ {fps:.2f}fps  {w}x{h}")

    KsA = A["K"][seg].copy() * args.scale
    KsA[:, 2, 2] = 1.0
    KsB = B["K"][seg].copy() * args.scale if B is not None else None
    if KsB is not None:
        KsB[:, 2, 2] = 1.0

    # 世界系取景：以 A 组骨盆水平轨迹中心为原点；相机放到身体正前方。
    # 用**这一段**来定取景（而不是整片）：渲染某一段时本来就是看这一段。
    grid_center = (0.0, 0.0)
    cam_facing = np.array([0.0, 0.0, 1.0])
    ref = gA if gA is not None else gB
    if ref is not None:
        rj = ref[1][seg]
        grid_center = (float(rj[:, 0, 0].mean()), float(rj[:, 0, 2].mean()))
        fs = np.stack([body_facing(rj[i]) for i in range(0, count, max(1, count // 30))])
        cam_facing = fs.mean(axis=0)
        nc = np.linalg.norm(cam_facing)
        cam_facing = cam_facing / nc if nc > 1e-6 else np.array([0.0, 0.0, 1.0])

    if not shutil.which("ffmpeg"):
        print("找不到 ffmpeg，无法编码输出。", file=sys.stderr)
        return 1

    cam_side = np.cross(np.array([0.0, 1.0, 0.0]), cam_facing)
    _ns = np.linalg.norm(cam_side)
    cam_side = cam_side / _ns if _ns > 1e-6 else np.array([1.0, 0.0, 0.0])

    grid = parse_grid(args.grid)
    cell_h = args.cell_h or h

    audio_in = None
    if args.audio:
        audio_in = Path(args.audio).resolve()
        if not audio_in.exists():
            print(f"找不到音轨来源：{audio_in}", file=sys.stderr)
            return 1
        has_audio = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(audio_in)],
            capture_output=True, text=True).stdout.strip()
        if not has_audio:
            print(f"该文件没有音轨，忽略 --audio：{audio_in}")
            audio_in = None

    # 视频走 stdin 裸帧管道；音轨另开一路输入直接 copy（不重编码）。
    # 必须显式 -map：不指定的话 ffmpeg 的默认流选择不一定挑中音轨。
    # -shortest 兜底——60->30 降帧后画面与音轨长度可能差一两帧。
    # 画布尺寸得**先渲一帧**才知道：格数×原片宽那种算法在多格/缩放/以后的
    # 特写格（裁剪出正方形）下都不成立。
    ctx = {
        "faces": faces, "cells": cells, "grid": grid, "cell_h": cell_h,
        "compare": layout == "compare", "alpha": args.alpha,
        "w": w, "h": h, "cam_facing": cam_facing, "cam_side": cam_side,
        "grid_center": grid_center, "grid_step": args.grid_step,
        "grid_extent": args.grid_extent,
    }
    build = make_frame_builder(ctx)

    def pack_frame(frame, i):
        """第 i 帧（段内序号）的输入。帧从 FrameSource 顺序拿，见 pack。"""
        per = {"frame": frame, "verts": A["verts"][start + i],
               "joints": A["joints"][start + i], "K": KsA[i],
               "gv": gA[0][start + i] if gA is not None else None,
               "gj": gA[1][start + i] if gA is not None else None}
        if compare:
            per.update({
                "b_verts": B["verts"][start + i] if B is not None else None,
                "b_joints": B["joints"][start + i] if B is not None else None,
                "b_gv": gB[0][start + i] if gB is not None else None,
                "b_gj": gB[1][start + i] if gB is not None else None,
            })
        return per

    first_canvas = build(pack_frame(first, 0))
    out_h, out_w = first_canvas.shape[:2]
    print(f"画布: {out_w}x{out_h}（{len(cells) or 4} 格，每格高 {cell_h}）")

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-"]
    if audio_in is not None:
        if start > 0:
            # 输入侧 seek：只渲一段时音轨也要从那一段起，否则声画错位
            cmd += ["-ss", f"{start / fps:.6f}"]
        cmd += ["-i", str(audio_in)]
    cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if audio_in is not None:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"]
    cmd += [args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def sink(buf):
        proc.stdin.write(buf)

    def pack(i):
        """第 i 帧之后那一帧（i 是**相对已写出的帧数**，不是段内序号）。

        按递增 i 调用（render_parallel 的分发循环就是递增的），所以这里能直接
        从 FrameSource 顺序取；取不到就是视频比 npz 短，返回 None 让上层收尾。
        """
        f = src.read()
        return None if f is None else pack_frame(f, i + 1)

    workers = args.workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    # Windows 用 spawn 起子进程，子进程要按 __main__.__file__ 重新导入主模块。
    # 脚本若是从管道喂给 python（`python - <<EOF`、`python -c`），__file__ 是
    # `<stdin>`，子进程会 OSError 起不来——旧版会表现成永久卡住。宁可慢，不要静默挂死。
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if workers > 1 and (not main_file or not Path(main_file).is_file()):
        print(f"⚠ 主程序不是磁盘上的文件（__file__={main_file!r}），"
              "多进程无法启动，已退回单进程。请以文件方式运行本脚本。", file=sys.stderr)
        workers = 1

    # 第 0 帧（就是上面那帧尺寸探针）直接写出，剩下的从第 1 帧起交给并行。
    sink(first_canvas.tobytes())
    rest = count - 1
    if workers > 1 and rest > 1:
        print(f"并行渲染：{workers} 个进程")
        render_parallel(rest, pack, ctx, (out_h, out_w, 3), workers, sink, out_w, out_h)
    else:
        for i in range(rest):
            per = pack(i)
            if per is None:
                break
            sink(build(per).tobytes())
            if (i + 1) % 25 == 0 or i + 1 == rest:
                print(f"  渲染 {i + 2}/{count}", end="\r", flush=True)

    proc.stdin.close()
    proc.wait()
    size = Path(args.out).stat().st_size / 1024 / 1024
    print(f"\n完成: {args.out}  ({size:.1f} MB)")
    if layout == "compare":
        print(f"四格对比：原片 | 两组叠加({COLOR_A['label']}蓝 / {COLOR_B['label']}橙) "
              "| 世界系正面 | 世界系侧面")
        print("看什么：两色重合 = 一致、可信；两色分开 = 那一段就是不确定度所在。")
        print("       差异 92% 是整体平移，所以你会看到两个 mesh 整体错开约 7cm。")
    else:
        desc = " | ".join(
            c["kind"] + (f"@{c['yaw']:g}°" if c["kind"] == "world" else "")
            for c in cells)
        print(f"{len(cells)} 格：{desc}（世界系网格间距 {args.grid_step}m）")
        print("提示：红色是骨盆区域。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
