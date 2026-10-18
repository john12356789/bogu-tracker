"""把重建出的骨盆 6 自由度动作转成 funscript（供 OSR2+/SR6 等多轴设备回放）。

为什么是 6 轴对 6 轴
-------------------
骨盆本身就是一个 6 自由度刚体（3 平移 + 3 旋转），SR6 有 6 个轴，所以映射
几乎是天然的 1:1，**不需要写 Stewart 平台逆运动学**——固件自己做 IK，主机只发 6 个
归一化轴值。

轴对应（TCode v0.3 规范里的正方向定义，均相对"使用者"）
    L0 上/下        <- 骨盆竖直平移
    L1 远离使用者   <- 骨盆前后平移
    L2 使用者左侧   <- 骨盆左右平移
    R0 绕 L0 扭转   <- 骨盆绕竖直轴旋转 = 扭胯
    R1 绕 L1 翻滚   <- 骨盆绕前后轴旋转
    R2 绕 L2 俯仰   <- 骨盆绕左右轴旋转

**L0 是竖直方向，而 GVHMR 的世界系本来就是重力对齐的**（gravity-view 方法的核心），
所以竖直轴不需要额外标定。

两个关键设计决定
----------------
1. **不平移世界系轨迹，而是高通后再用**。世界系轨迹包含走位（舞者在台上移动几米），
   设备复现"向左走了三米"是错的；而且本项目已实测：相机系高度可靠，**不确定度全在
   世界系轨迹环节**（相机系分歧 1.5cm vs 世界系 6.8cm）。运动平台行业的惯例也是
   对平移通道做高通——平台行程有限，只要瞬态不要绝对位置。用 `--highpass 0` 关闭。
2. **旋转从关节位置构造，不从轴角参数转**。本项目在旋转表示上踩过坑
   （`matrix_to_rotation_6d` 取前两行而非列、四元数按最大分量而非 w>=0 规范化）。
   直接由 pelvis/双胯/spine1 三点构正交基，全程无表示转换。

必须知道的两件事
----------------
- **幅度收缩（去偏后）**：直接拿重建幅度比关键点幅度，比值会被**系统性低估**——
  OLS 斜率的自变量（关键点）带噪声时被压向零（回归稀释）。修正 `k = 斜率/r²` 后，
  本项目 5 条素材的保留率收敛到 **0.60-0.87（中位 0.67）**，即约三分之一被压扁，
  而不是原始比值显示的 0.12-0.54。用 `--de-bias` 按 1/k 补齐。
- **欧拉分解顺序是一个真实的选择**：从完整 SO(3) 里取 3 个角，不同顺序给出不同角值，
  幅度大时还会接近万向锁。默认 `YZX`（对应 R0->R1->R2），可用 `--order` 改。

用法
----
    python src/to_funscript.py <incam_npz> --out funscript/ --video <processed.mp4>
    python src/to_funscript.py <incam_npz> --out funscript/ --highpass 0 --travel-l0 0.12
    python src/to_funscript.py <incam_npz> --out funscript/ --fit-window 5   # 按时间窗归一化

跨切镜头
--------
素材剪过镜头时，世界系的绝对位置跨切点不可观测（重建会重新定位），这个台阶经
一阶高通之后是一次满行程撞击，而且会污染 --fit 的分位、把全片幅度压小。切点先由
`src/scene_cuts.py` 测出来，再交给这里处理：

    python src/scene_cuts.py <processed.mp4> --out cuts.json
    python src/to_funscript.py <npz> --out funscript/ --cuts cuts.json --cut-reset

**不给 --cuts 时这套代码一行都不走**，输出与没有这个功能时逐字节一致（有测试
盯着，见 tests/test_scene_cuts.py）。--cut-reset 与 --cut-blank 各自独立开关，
--cut-scale 是占位，没有实现（原因见该参数的说明）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from viz import sources  # noqa: E402
from viz.schema import MotionSequence, project_points  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

# TCode 轴 -> 该轴的物理来源说明。顺序即输出顺序。
AXES = ["L0", "L1", "L2", "R0", "R1", "R2"]
AXIS_DESC = {
    "L0": "上下（骨盆竖直平移）",
    "L1": "前后（骨盆前后平移）",
    "L2": "左右（骨盆左右平移）",
    "R0": "扭转（绕竖直轴 = 扭胯）",
    "R1": "翻滚（绕前后轴）",
    "R2": "俯仰（绕左右轴）",
}
# 每个次要轴对应的 funscript 文件名后缀（MultiFunPlayer 按名字自动分区）
SUFFIX = {"L0": None, "L1": "surge", "L2": "sway",
          "R0": "twist", "R1": "roll", "R2": "pitch"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """argv=None 时读 sys.argv（CLI）。Web 后端传一个列表进来，
    这样默认值、类型校验、取值范围都只有一份定义，不会和 CLI 漂移。"""
    p = argparse.ArgumentParser(
        description="骨盆 6 自由度 -> funscript（多轴）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", help="hmr4d_results_incam.npz（内含 cam_to_world，可转到世界系）")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--video", default=None, help="处理后的视频，用来确定帧率")
    p.add_argument("--fps", type=float, default=None, help="覆盖帧率（默认从 npz 或视频读取）")
    p.add_argument("--name", default=None, help="输出文件名主干（默认用 npz 的父目录名）")
    p.add_argument("--highpass", type=float, default=0.5,
                   help="平移通道的高通截止频率 Hz（默认 0.5；0=不平移，直接用世界系轨迹）。"
                        "去除走位/漂移，只保留瞬态——平台行程有限，这也是行业惯例。"
                        "**有用区间 0.5~1Hz**：0.2Hz 的走位在 0.5Hz 处只削 −9dB、"
                        "1Hz 处削 −15dB，而 1Hz 的扭胯在 1Hz 处只掉 −3.9dB —— 再往上"
                        "（比如 6Hz）**削掉的是动作本身**（1Hz 的扭胯只剩 10%），"
                        "只会让变频后的输出更小更抖，不会让高频更灵敏。"
                        "要「高频更跟手」就改「平滑」：它才是掐高频的那个")
    p.add_argument("--highpass-rot", type=float, default=None, metavar="HZ",
                   help="旋转通道的高通截止频率（默认与 --highpass 相同；0=关闭）。"
                        "有用区间与平移同（0.5~1Hz），但**不能太低**：它的主要职责是"
                        "去掉累积转向，截止频率低于 0.3Hz 就去不干净。"
                        "必要的原因：重建的骨盆偏航可能累积成一整圈甚至多圈（本项目实测某条"
                        "素材达 686°），而 R0 的行程有限，直接解环绕再适配会把真正的扭胯压平。"
                        "运动模拟器对朝向也是这么处理的——只复现摆动，不复现累积转向")
    p.add_argument("--travel-l0", type=float, default=0.08, metavar="M",
                   help="L0 的半行程（米），默认 0.08")
    p.add_argument("--travel-l1", type=float, default=0.08, metavar="M", help="L1 的半行程（米）")
    p.add_argument("--travel-l2", type=float, default=0.08, metavar="M", help="L2 的半行程（米）")
    p.add_argument("--max-deg", type=float, default=45.0,
                   help="旋转半步程（度），默认 45。R0/R1/R2 共用")
    p.add_argument("--neutral", choices=["mean", "first"], default="mean",
                   help="中立位定义：整段均值（默认）或首帧")
    p.add_argument("--reference", choices=["pelvis", "genital"], default="pelvis",
                   help="跟随的参考点。pelvis=骨盆关节本身（默认）；genital=骨盆前下方的接触点。"
                        "两者共用同一套姿态（接触点在刚体上，朝向就是骨盆朝向），"
                        "差别只在**平移**：接触点离旋转中心有一根杆臂，"
                        "骨盆一转动它就会多出一段位移。见 --ref-* 的说明")
    p.add_argument("--ref-fwd", type=float, default=0.08, metavar="M",
                   help="接触点在骨盆局部系里的**前方**偏移（米，正=朝身体前方）。"
                        "默认 0.08 是耻骨联合相对骨盆关节中心的大致前后距离，"
                        "**属于人体测量学估值，应当按实际佩戴位置标定**")
    p.add_argument("--ref-down", type=float, default=0.09, metavar="M",
                   help="接触点在骨盆局部系里的**下方**偏移（米，正=朝脚下）。"
                        "默认 0.09 同样是估值，应按实际佩戴位置标定")
    p.add_argument("--order", default="YZX",
                   help="欧拉分解顺序（intrinsic）。局部轴 x=右 y=上 z=前，"
                        "所以 YZX = 先绕上(R0) 再绕前(R1) 再绕右(R2)。默认 YZX")
    p.add_argument("--flip", default="", help="翻转指定轴的符号，逗号分隔，如 R0,R2")
    p.add_argument("--smooth", type=int, default=2,
                   help="对输出做 N 点滑动平均（默认 2）。机械需要限 jerk，0=关闭")
    p.add_argument("--fit", type=float, default=99.0, metavar="P",
                   help="自动适配行程：按 |值的 P 百分位映射到半步程，默认 99（即裁剪 <1%%）。"
                        "0 = 关闭，直接按 --travel-*/--max-deg 裁剪（实测会削掉三成以上）")
    p.add_argument("--fit-window", type=float, default=0.0, metavar="SEC",
                   help="按时间窗归一化（**独立的第二段**）：每帧的尺度取该帧 ±SEC/2 秒窗内 "
                        "|值| 的分位，把该分位映射到行程极限，而不是全段共用一个。"
                        "0 = 关闭（默认）。\n"
                        "为什么需要：全程归一化时**一个大幅动作就把比例尺定死**，"
                        "安静段落被压成近似平线。\n"
                        "与 --fit 无关：--fit 关着时本段自取 99%% 分位，照样生效；"
                        "本段开启时 --travel-* / --max-deg 反过来不再影响输出"
                        "（尺度按窗内分位定，行程设定用不上）。\n"
                        "**代价必须知道**：它改写的是动作的**相对幅度**。幅度不再可比——"
                        "安静段落被抬起来，设备做的事比人大。要保留真实幅度关系就别开。")
    p.add_argument("--de-bias", action="store_true",
                   help="幅度去偏校正：把动作按 1/k 放大，k 由误差变量模型现算。\n"
                        "为什么需要：直接拿重建幅度比关键点幅度，得到的比值被系统性低估——"
                        "OLS 斜率的自变量（关键点）带噪声时会被压向零（回归稀释）。"
                        "修正 k = 斜率/r²，该量在估计误差下无偏。实测 5 条素材原始比值相差 4.5 倍，"
                        "修正后收敛到 0.60–0.87，说明确实存在约三分之一的系统性压缩。\n"
                        "**注意**：这是让幅度在统计上匹配估计真值，不会让 2D 拟合变好"
                        "（2D 残差主要来自关键点噪声）")
    # ---- 切点处理（跨镜头）----
    #
    # 这四个是**独立开关**，默认全关：不给 --cuts 时这套代码一行都不走，输出与
    # 没有这个功能时逐字节一致。理由——跨切处理会改动已提交产物，不该在别人
    # 不知情的时候改变结果。
    p.add_argument("--cuts", default=None, metavar="JSON",
                   help="切点清单（src/scene_cuts.py 出的 json）。**不给就完全不走"
                        "跨切处理**。\n"
                        "为什么需要：跨过一次硬切，世界系的绝对位置就不可观测了——"
                        "GVHMR 的世界系原点与朝向由相机轨迹推出来，镜头一换就重新定位，"
                        "于是同一个静止的人也会\"瞬移\"。这个台阶进了 funscript 就是"
                        "一次满行程撞击，而且还会污染 --fit 的分位统计（把全片比例尺"
                        "抬高、真实动作被压小）。")
    p.add_argument("--cut-reset", action="store_true",
                   help="在切点做去阶跃（见 highpass 的说明）。切点处的落差恒为 0，"
                        "代价是之后约 0.3s 偏离真实交流分量（单调回正，不是跳变）")
    p.add_argument("--cut-blank", type=float, default=0.0, metavar="SEC",
                   help="切点**之后** SEC 秒（含切点那一帧）标为无效，用两端有效样本"
                        "线性插值填上。0=关闭（默认，这套功能整体默认关）。\n"
                        "切点之前不动：那几帧仍是上一条镜头的有效观测。\n"
                        "它管的是另一件事：切点的模型时间上下文来自**另一条镜头**，"
                        "那几帧的估计本身就是垃圾，掩蔽把它们换掉；顺带把 --cut-reset "
                        "回正的那 0.3s 插直。\n"
                        "多宽：0.2 是个稳妥起点；调大更干净，代价见报告里"
                        "\"掩蔽吃掉 N 帧\"那一行")
    p.add_argument("--cut-scale", action="store_true",
                   help="【未实现，留作占位】按镜头分别估计焦距，消除景别差异带来的"
                        "幅度台阶。\n"
                        "为什么没做：实测这条路在这条管线上**没有信号**。内参硬编码成"
                        "53.13° 对角，若某条镜头的真实焦距不同，重建会把这个误差吸收进"
                        "**深度**（人被放到更远/更近处），而身体的度量尺寸由 SMPL-X 的"
                        "体型先验钉住——实测本素材 7 段之间的骨长只差 0.3%~2.9%（胯宽"
                        "0.92%、腰长 0.31%），而 3D/2D 比值那 70% 的变化完全由重建自己"
                        "选的深度解释得通。这是单目度量尺度不可观测的老问题，不是实现"
                        "不到位。\n"
                        "要做只能引外部信息：预训练的单图 FOV 模型（Perspective Fields "
                        "等），或人工按预览给每段一个倍数。")
    # ---- 后处理（输出前）----
    #
    # 这两项都作用在**归一化之后的 0~1 数组**上，也就是写完就是 funscript 的那一步。
    # **默认全关**：不开的话输出与以前逐元素相同（回归由 test_postprocess 盯着）。
    p.add_argument("--height-limit", action="store_true",
                   help="后处理 A：抬高时把 L1/L2 向中立收拢，越高压得越紧。\n"
                        "物理上的理由：重心越高，同样的横摆越接近倾覆，机器自身的"
                        "横向行程也越紧张。参数见 --hl-start / --hl-top。")
    p.add_argument("--hl-start", type=float, default=0.5, metavar="0~1",
                   help="后处理 A：从多高开始收（**脚本自己的 L0 刻度**，0.5 = 中立位）。\n"
                        "用脚本刻度而不是物理高度：所见即所得，代价是 L0 削顶时"
                        "\"高度\"也跟着到顶（削顶比例在报告里）。")
    p.add_argument("--hl-top", type=float, default=0.0, metavar="0~1",
                   help="后处理 A：抬到顶时**保留**的行程倍率。0 = 完全居中（默认），"
                        "1 = 等于不生效。介于两者之间就是\"高处只留一点横摆\"。")
    p.add_argument("--compensate", action="store_true",
                   help="后处理 B：用位移补偿旋转——横向/前后走了多远，就用对应的"
                        "旋转反向补回来，把重心转回支撑面之上。\n"
                        "两条耦合：前后 L1 → pitch R2、左右 L2 → roll R1。"
                        "增益见 --comp-pitch / --comp-roll。")
    p.add_argument("--comp-pitch", type=float, default=-0.5, metavar="K",
                   help="后处理 B：前后位移（L1）对俯仰（R2）的增益。负号 = 反向抵消"
                        "（向前走了就向后倾）。0 = 这条耦合关掉。\n"
                        "单位：位移偏离中立 1.0（满刻度 ±0.5），旋转补 0.5×K。")
    p.add_argument("--comp-roll", type=float, default=-0.5, metavar="K",
                   help="后处理 B：左右位移（L2）对横滚（R1）的增益。负号 = 反向抵消"
                        "（向右走了就向左倾）。0 = 这条耦合关掉。")
    p.add_argument("--gain-l0", type=float, default=1.0, metavar="K",
                   help="后处理 C 逐轴倍乘：主轴（上下）的行程倍率。1 = 不变。")
    for _ax in ("l1", "l2", "r0", "r1", "r2"):
        p.add_argument(f"--gain-{_ax}", type=float, default=1.0, metavar="K",
                       help=f"后处理 C 逐轴倍乘：{_ax.upper()} 的行程倍率。1 = 不变，"
                            f"0 = 锁在中立位。")
    p.add_argument("--post-order", default=",".join(POST_STEPS), metavar="LIST",
                   help="后处理（输出前）的执行顺序，逗号分隔。界面按拖动排出来的"
                        "就是它。这几步互不交换（先限位再补偿，补的是限位之后的"
                        "行程），所以要能定序。认不出的 id 会被丢掉、没提到的补在"
                        "末尾。可选：" + " / ".join(POST_STEPS))
    p.add_argument("--person-timeline", default=None, metavar="JSON",
                   help="多人素材：按帧指定用**哪条轨迹**的重建结果，逐段拼成一条序列。\n"
                        "格式 [{\"from\":0,\"to\":3332,\"npz\":\"…/hmr4d_results_incam.npz\"}, …]，"
                        "区间必须从第 0 帧起、首尾相接、不重叠；npz 省略时用位置参数那一份。\n"
                        "**接缝会并进 --cuts**：两个人在各自的世界系里重建，接缝处的绝对"
                        "位置不可比——这和镜头切换是同一类问题，去阶跃/掩蔽那套机器直接用。")
    p.add_argument("--range", default=None, metavar="A:B",
                   help="只算这一段（帧号，左闭右开）。\n"
                        "用途是**每段独立成篇**：多人素材里按人物把片子切成几段，"
                        "每段各自跑一遍归一化，再把归一化后的值拼起来。\n"
                        "写出的 funscript **时间戳是绝对帧号**（从 A 起），所以几段的"
                        "产物可以直接按时间拼。")
    p.add_argument("--rate", type=float, default=None,
                   help="输出采样率 Hz（默认与视频一致）。不要随意上调——插值帧是合成的")
    return p.parse_args(argv)


def pelvis_frames(J: np.ndarray) -> np.ndarray:
    """由关节位置构造逐帧骨盆姿态矩阵 (T,3,3)，列向量为 [右, 上, 前]（世界系）。

    用关节而不是轴角参数：不依赖任何旋转表示约定。三点构基再正交化。
    索引 0=pelvis 1=左胯 2=右胯 3=spine1（见 viz/schema.py）。
    """
    right = J[:, 2, :] - J[:, 1, :]
    up = J[:, 3, :] - J[:, 0, :]
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-9)
    up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    # 施密特正交化：up 扣掉沿 right 的分量
    up = up - (up * right).sum(axis=1, keepdims=True) * right
    up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    fwd = np.cross(right, up)          # 右手系：right × up = fwd
    return np.stack([right, up, fwd], axis=2)


def orthonormal(M: np.ndarray) -> np.ndarray:
    """极分解投影到 SO(3)：关节点估计出来的基不严格正交，直接喂 Rotation 会报错。"""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:           # 反射的话翻最后一个奇异向量
        U[:, -1] *= -1
        R = U @ Vt
    return R


def highpass(x: np.ndarray, fps: float, fc: float,
             cuts: list[int] | None = None) -> np.ndarray:
    """简单一阶高通：x - 低通(x)。够用，且不引入明显相位失真以外的东西。

    `cuts` 给出切点帧号时，在切点做**去阶跃**：把低通状态推到"输出保持不变"的
    位置。为什么需要——一阶高通对阶跃 S 的响应是峰值 `a·S`、时间常数 `1/(2π·fc)`
    的冲击（fc=0.5Hz、30fps 时 a=0.9006、τ=0.32s），也就是**几乎把整个台阶原样
    打出去**。跨镜头的世界系台阶动辄几十厘米，这就是"切镜头时脚本猛跳一下"的
    全部来源。

    去阶跃的取法是 `prev = x[i] - y[i-1]/a`：代进标准更新（`prev ← a·prev +
    (1-a)·x[i]`）后正好 `y[i] = y[i-1]`——切点处的落差**恒等于 0**，台阶被整个
    吸收进低通状态，之后以 τ 衰减回该段真正的交流分量，全程没有一帧跳。
    那个 `/a` 不能省：少了它得到的是 `y[i] = a·y[i-1]`，还留着 10% 的落差。
    实测（本素材 6 个切点取最大，半行程为单位）：L0 2.49→0.31，R0 0.49→0.02。

    **不要写成 `prev = x[i]`**（把输出按到 0，看起来更"干净"）。那等于在切点强行
    把轴拉到中立位：那一刻要是在做大幅度动作，落差就等于动作本身的幅度——
    实测 L0 反而从 2.49× 恶化到 3.15×，比不处理还差。
    """
    if fc <= 0:
        return x
    a = float(np.exp(-2.0 * np.pi * fc / fps))
    y = np.empty_like(x)
    prev = x[0]
    cutset = set(cuts) if cuts else frozenset()
    for i in range(len(x)):
        if i in cutset and i > 0:
            prev = x[i] - y[i - 1] / a
        prev = a * prev + (1 - a) * x[i]
        y[i] = x[i] - prev
    return y


def blank_cuts(x: np.ndarray, cuts: list[int] | None, after: int) -> int:
    """把每个切点**起**的 after 帧标为无效（含切点那一帧），用两端最近的有效样本
    线性插值填上（**原地改**）。

    返回被抹掉的样本数。after<=0 或没有切点时不动。

    为什么这一步不能省：切点前后模型看到的时间上下文来自**另一条镜头**，
    那几帧的估计本身就是垃圾。留着不是"多保留了一点信息"，是把噪声当成动作。

    **窗口只在切点之后**（2026-10-17 改；原来是 `[c-after, c+after]` 两侧对称）。
    切点那一帧是新镜头的第一帧，它和它之后的 after 帧，模型的时间上下文跨了两条
    镜头，估计不可用；而切点**之前**的帧仍然是上一条镜头的有效观测，没有理由动
    它们——对称窗白白盖掉一段真实动作，还让"切点前的曲线"跟着变。改成单侧之后
    左端锚点正好是切点前那一帧，插值从那里接出去，切点处的曲线照旧连续。

    顺序：本函数在**高通之后**调用（见调用点的说明）。先高通再掩蔽时，去阶跃
    已经把台阶吃掉了，掩蔽只需要把回正的那 0.3s 插直；反过来先掩蔽再高通的话，
    插值桥把原始台阶摊成一段斜坡，斜坡的低频照样漏得出去。
    """
    if not cuts or after <= 0:
        return 0
    n = len(x)
    bad = np.zeros(n, dtype=bool)
    for c in cuts:
        bad[max(0, c): min(n, c + after + 1)] = True
    if not bad.any():
        return 0
    good = ~bad
    if good.sum() < 2:                      # 整段都被切点盖住，没什么可插的
        return int(bad.sum())
    idx = np.arange(n)
    if x.ndim == 1:
        x[:] = np.interp(idx, idx[good], x[good])
    else:
        for k in range(x.shape[1]):
            x[:, k] = np.interp(idx, idx[good], x[good, k])
    return int(bad.sum())


def cut_steps(x: np.ndarray, cuts: list[int] | None) -> list[float]:
    """每个切点处的原始台阶大小（带符号），用来在报告里说明"本来会打出去多少"。"""
    if not cuts:
        return []
    return [float(x[c] - x[c - 1]) for c in cuts if 0 < c < len(x)]


def cut_jump(x: np.ndarray, cuts: list[int] | None) -> float:
    """切点处**那一帧的落差** |x[c] − x[c−1]| 的最大值。

    这就是"突变"本身，也正是设备挨的那一下。**不要用切点邻域的最大绝对偏离**：
    那个量的主体是真实动作（本素材原始 L0 本身就有 ±0.3 m 的幅度），
    处理前后几乎不变，看不出区别——实测把 292.9mm 对 292.9mm 报成"没改善"，
    而同一份数据里切点处的落差其实从 41mm 降到了 0。
    """
    if not cuts:
        return 0.0
    n = len(x)
    return max((abs(float(x[c] - x[c - 1])) for c in cuts if 0 < c < n), default=0.0)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = np.concatenate([np.repeat(x[:1], k // 2), x, np.repeat(x[-1:], k // 2)])
    return np.convolve(pad, np.ones(k) / k, mode="same")[k // 2:k // 2 + len(x)]


# 后处理（输出前）的**步骤及其默认执行顺序**。
#
# 顺序是数据不是代码：界面可以拖动排序（--post-order），因为这几步互相不交换
# ——"先限位再补偿"和"先补偿再限位"补的不是同一个行程。id 与前端一一对应。
POST_STEPS = ("height_limit", "compensate", "gain")
POST_STEP_NAME = {"height_limit": "高度相关限位", "compensate": "位移→旋转补偿",
                  "gain": "逐轴倍乘"}
GAIN_AXES = ("L0", "L1", "L2", "R0", "R1", "R2")


def parse_post_order(spec: str | None) -> list[str]:
    """``--post-order`` 的字符串 -> 步骤 id 列表。

    **认不出的丢掉、没提到的补在末尾**：顺序是界面拖出来的，但后端不该因为一个
    拼错的 id 就少跑一步——那会静默改掉输出，而用户以为那个开关还开着。
    """
    out: list[str] = []
    for x in str(spec or "").replace("，", ",").split(","):
        x = x.strip()
        if x in POST_STEPS and x not in out:
            out.append(x)
    return out + [x for x in POST_STEPS if x not in out]


def postprocess(norm: dict[str, np.ndarray], order: list[str],
                on_height: bool, hl_start: float, hl_top: float,
                on_comp: bool, comp_pitch: float, comp_roll: float,
                gains: dict[str, float] | None = None
                ) -> tuple[dict[str, np.ndarray], dict]:
    """输出前那几个后处理，**按 order 依次作用**。输入输出都是 0~1（0.5 = 中立）。

    逐步做什么、为什么，见各自的 _step_*；这里只管"按什么顺序作用"——这几步
    互相不可交换（先限位再补偿，补的是限位之后的行程），所以顺序是个参数。

    关掉的步骤不作用；`gains` 全为 1 时逐轴倍乘也**逐位不动**（回归线见
    src/tests/test_postprocess.py）。
    """
    out = {k: np.asarray(v, dtype=float).copy() for k, v in norm.items()}
    done: list[str] = []
    stats: dict = {}
    for step in order:
        if step == "height_limit" and on_height:
            _step_height(out, hl_start, hl_top, stats)
            done.append(step)
        elif step == "compensate" and on_comp:
            _step_compensate(out, comp_pitch, comp_roll, stats)
            done.append(step)
        elif step == "gain" and any(abs(float(v) - 1.0) > 1e-12
                                    for v in (gains or {}).values()):
            _step_gain(out, gains or {}, stats)
            done.append(step)
    stats["order"] = done            # 实际跑过的，按执行顺序（报告里照这个念）
    return out, stats


def _step_height(out: dict, start: float, top: float, stats: dict) -> None:
    """A 高度相关限位：抬得越高，L1/L2 越向中立收。

    物理上的理由：重心抬得越高，同样的横摆越接近倾覆，机器自身的横向行程也越紧张。
    「高度」用的是**脚本自己的 L0**（0~1），不是物理高度——所见即所得，代价是
    L0 削顶时"高度"也跟着到顶（削顶比例在报告里）。
    """
    g = 1.0 - (1.0 - float(np.clip(top, 0.0, 1.0))) * np.clip(
        (out["L0"] - start) / max(1e-6, 1.0 - start), 0.0, 1.0)
    # 只在真正开始收的帧上改写（g < 1）。一律套公式的话，g 恰好等于 1 的那些帧
    # 也会被 0.5+(x-0.5)*1 洗一遍，浮点上差个 1e-17——数值上无所谓，但"低于
    # 起始高度一动不动"这句话就不再字面成立，回归测试也没法用逐元素相等盯住。
    for k in ("L1", "L2"):
        out[k] = np.where(g < 1.0, 0.5 + (out[k] - 0.5) * g, out[k])
    stats["height_limit"] = {"mean_g": float(g.mean()), "min_g": float(g.min()),
                             "start": start, "top": top}


def _step_compensate(out: dict, k_pitch: float, k_roll: float, stats: dict) -> None:
    """B 位移→旋转补偿：横向/前后走了多远，就用对应的旋转反向补回来。

    默认增益为负：向右走了就往左倾，把重心转回支撑面之上。**补的是当时的行程**
    ——所以它在顺序里排在哪，就补哪个阶段的位移（见 --post-order）。

    增益的物理值取决于两个轴各自的行程标定（真正的物理关系是 θ = atan(d/h)，
    需要米数和真实高度），从脚本刻度里推不出来，默认值只是起点。
    """
    got = {}
    for axis, src, k in (("R2", "L1", k_pitch), ("R1", "L2", k_roll)):
        if axis not in out or src not in out or not k:
            continue
        add = float(k) * (out[src] - 0.5)
        raw = out[axis] + add
        out[axis] = np.clip(raw, 0.0001, 0.9999)
        got[axis] = {"src": src, "gain": float(k),
                     "mean_shift": float(np.abs(add).mean()),
                     "clipped": float(np.mean((raw < 0) | (raw > 1)))}
    stats["compensate"] = got


def _step_gain(out: dict, gains: dict[str, float], stats: dict) -> None:
    """C 逐轴倍乘：每条轴各自乘一个系数（1 = 不变，0 = 该轴锁在中立位）。

    和别的步骤一样只缩放"离中立多远"，不动中立位本身。排在顺序最后就是"整体
    再修一遍幅度"；排在最前则是先定各轴的相对权重，后面的步骤再作用。
    """
    got = {}
    for k, g in (gains or {}).items():
        if k not in out or abs(float(g) - 1.0) <= 1e-12:
            continue
        raw = 0.5 + (out[k] - 0.5) * float(g)
        out[k] = np.clip(raw, 0.0001, 0.9999)
        got[k] = {"gain": float(g),
                  "clipped": float(np.mean((raw < 0) | (raw > 1)))}
    if got:
        stats["gain"] = got


# 按时间窗归一化时，窗内尺度相对全段尺度的**下限**（等价于增益上限 MAX_BOOST 倍）。
#
# 窗内几乎静止（只剩跟踪噪声）时 q_win -> 0，增益会冲到几十上百倍，把噪声放大成
# 满行程的往复——**这对真机是危险动作**，而且它驱动的是一条"人没动"的曲线。
# 所以给增益封顶：安静的段落可以被抬起来，但抬不过 MAX_BOOST 倍。
MAX_BOOST = 3.0


def window_scale(a: np.ndarray, fps: float, win_s: float, pct: float) -> np.ndarray:
    """逐帧的归一化尺度 q(t) = |v| 在 t±win/2 秒窗内的 pct 分位。(T,)

    ## 为什么不是 FSP 那版的「窗内 min/max」

    FSP 的 `actionsFromSignal` 用的是窗内 min/max 映射到 0..1，那等价于**同时重定
    中心和缩放**：每个窗都被拉到满行程，输出就不再以 0.5 为中立位了。而设备中点
    是有物理意义的（0.5 = 行程正中），按窗重定中心会让设备随窗漂移——人在某个
    姿势上静止，设备反而停在那个窗的偏置上。

    这里只改**增益**、不动**中心**：u = 0.5 + 0.5·v/q(t)，中心永远是 v=0
    （由 --neutral 定的中立姿势）。于是它是现有全局分位归一化的严格推广——
    win_s -> 无穷时 q(t) 退化成常数，就是原式。

    ## 为什么用分位而不是窗内 max

    与 --fit 的语义保持一致（按 |值的 P 分位映射到半步程），开不开窗只差
    "分位是全局算还是逐帧算"，其余不受影响。

    窗口强制取奇数并**以当前帧为中心**：否则增益会带半个窗的滞后，
    快动作上表现为设备比画面对不上。
    """
    w = max(3, int(round(win_s * fps)) | 1)          # |1 保证是奇数
    if w >= a.size:
        # 窗比素材还长 = 每个窗都装着整段，"局部"就没有意义了，直接退化成全程。
        # 这一条也是为了让 win -> 无穷的极限**精确**等于原式：否则边缘延拓会把
        # 首尾两帧在窗里反复计入（窗长 2 倍素材时有过半样本是这两帧的副本），
        # 分位数被它们带偏，极限值就不再是不开窗的结果了。
        return np.full(a.size, float(np.percentile(a, pct)))
    half = w // 2
    pad = np.concatenate([np.repeat(a[:1], half), a, np.repeat(a[-1:], half)])
    win = np.lib.stride_tricks.sliding_window_view(pad, w)   # (T, w)
    return np.percentile(win, pct, axis=1)


# --fit = 0（全程归一化关着）时，窗口阶段自己用的分位。
# 分位在这里的含义是"允许多少削顶"，两段问的是同一个问题，所以 --fit 开着时
# 窗口阶段沿用它的分位；关着时就用这个惯例值 99。
WIN_PCT_FALLBACK = 99.0


def window_normalize(d: np.ndarray, fps: float, win_s: float, pct: float,
                     env_k: int) -> tuple[np.ndarray, float, float, float]:
    """阶段 2：把 |d| 在每个时间窗内的 pct 分位映射到 1.0。返回 (d', gmin, gmax, capped)。

    **d 以"整个行程"为单位**：1.0 = 恰好走到行程极限。用这个单位是刻意的——
    换进来之后这个阶段就完全不知道行程是几米几度，于是它与全程缩放（阶段 1）、
    与 --travel-* / --max-deg 都无关，可以单独开、单独关。界面上的两个控件因此
    是独立的；行程滑杆在本阶段开启时失效（半行程在 d 里已经约掉了）。

    它是一个纯后处理：输入什么尺度都行，只按窗内幅度重新定标。全程归一化开着时，
    输出与"阶段 1 直接按窗内分位缩放"逐位相同——两种写法在数学上是一回事。

    返回的 gmin/gmax 是**相对全程的增益**（1.0 = 该帧与不开窗时同尺度），
    不是 d 上直接乘的那个系数。两者差一个常数 1/q_ref，混用会让人看不懂报告：
    --fit 关着时 q_ref 不等于 1，绝对值形式的"增益"会不以 1.0 为中心。
    """
    a = np.abs(d)
    q = smooth(window_scale(a, fps, win_s, pct), env_k)
    # 增益封顶的参照是"完全不开窗时会给的尺度"，也就是全段分位算出来的那个。
    q_ref = float(np.percentile(a, pct))
    floor = q_ref / MAX_BOOST if q_ref > 1e-9 else 0.0
    capped = float(np.mean(q < floor)) if floor > 0 else 0.0
    q = np.maximum(np.maximum(q, floor), 1e-9)
    # 输出的尺度除的是 q（窗内分位要映射到 1.0）；报告的增益除的是 q_ref
    # （要相对全程）。两者差一个常数，别合并。
    gain_rel = q_ref / q                # 1.0 = 与全程归一化给出的尺度相同
    return d / q, float(gain_rel.min()), float(gain_rel.max()), capped


def amplitude_report(seq) -> dict:
    """幅度保真度：重建的胯部动作幅度 vs 2D 观测幅度。

    做法与诊断脚本一致——把重建的左右胯投影回 2D，与 RTMPose 的关键点比，
    去掉慢趋势后取标准差，再算回归增益。增益 1.0 = 幅度被完整保留。
    """
    J = np.asarray(seq.joints)
    kp = seq.keypoints_2d
    n = seq.n_frames
    sp_m = np.empty(n)
    cen_m = np.empty((n, 2))
    for t in range(n):
        uv, _ = project_points(J[t, [1, 2]], seq.k_matrix(t), seq.world_to_cam(t))
        sp_m[t] = np.linalg.norm(uv[0] - uv[1])
        cen_m[t] = uv.mean(axis=0)
    sp_k = np.linalg.norm(kp[:, 11, :] - kp[:, 12, :], axis=1)
    cen_k = (kp[:, 11, :] + kp[:, 12, :]) / 2.0

    def dt(x, win=15):
        pad = np.concatenate([np.repeat(x[:1], win // 2), x, np.repeat(x[-1:], win // 2)])
        return x - np.convolve(pad, np.ones(win) / win, mode="same")[win // 2:win // 2 + len(x)]

    a, b = dt(sp_k), dt(sp_m)
    gain_twist = float(np.polyfit(a, b, 1)[0]) if a.std() > 1e-6 else float("nan")
    r_twist = float(np.corrcoef(a, b)[0, 1])
    ax = np.hypot(dt(cen_k[:, 0]), dt(cen_k[:, 1]))
    bx = np.hypot(dt(cen_m[:, 0]), dt(cen_m[:, 1]))
    gain_tran = float(np.polyfit(ax, bx, 1)[0]) if ax.std() > 1e-6 else float("nan")
    return {"gain_twist": gain_twist, "r_twist": r_twist, "gain_trans": gain_tran}


def estimate_retention(seq) -> tuple[float, float]:
    """去偏估计幅度保留率 k，返回 (旋转用, 平移用)。

    推导：设真实动作 a_true，关键点观测 a = a_true + 噪声 n，重建信号 b = k_ret·a_true。
    则 OLS 斜率(b~a) = k_ret·σ²_true/σ²_a = k_ret·r²，所以 **k_ret = 斜率 / r²**，
    与噪声量无关。直接看斜率会把压缩高估——实测把 67% 的保留率报成了 28%。

    两个通道分别估：左右胯跨距（扭胯主要体现在这里）与胯中点位移。
    """
    J = np.asarray(seq.joints)
    kp = seq.keypoints_2d
    n = seq.n_frames

    uv = np.empty((n, 2, 2))
    for t in range(n):
        u, _ = project_points(J[t, [1, 2]], seq.k_matrix(t), seq.world_to_cam(t))
        uv[t] = u

    def dt(x, w=15):
        """去趋势（15 帧滑均 ≈ 2Hz）。注意 mode="same" 返回的是填充后的长度，必须切回。"""
        r = w // 2
        pad = np.concatenate([np.repeat(x[:1], r), x, np.repeat(x[-1:], r)])
        return x - np.convolve(pad, np.ones(w) / w, mode="same")[r:r + len(x)]

    def retention(a_obs, b_rec):
        a, b = dt(np.asarray(a_obs, float)), dt(np.asarray(b_rec, float))
        if a.std() < 1e-6 or b.std() < 1e-6:
            return 1.0
        r = float(np.corrcoef(a, b)[0, 1])
        if r * r < 1e-9:
            return 1.0
        # clip 到 1.0：这个值只用来补齐，不该反过来压小已经很保守的重建
        return float(np.clip(float(np.polyfit(a, b, 1)[0]) / (r * r), 0.2, 1.0))

    m_span = np.linalg.norm(uv[:, 0] - uv[:, 1], axis=1)
    k_span = np.linalg.norm(kp[:, 11, :] - kp[:, 12, :], axis=1)
    m_mid = uv.mean(axis=1)
    k_mid = (kp[:, 11, :] + kp[:, 12, :]) / 2.0
    return (retention(k_span, m_span),
            retention(np.hypot(dt(k_mid[:, 0]), dt(k_mid[:, 1])),
                      np.hypot(dt(m_mid[:, 0]), dt(m_mid[:, 1]))))


def load_person_timeline(path: Path, primary: Path, fps: float):
    """读人物时间线，按帧拼成一条 MotionSequence。返回 (seq, 接缝帧号)。

    拼接发生在**姿态层**（逐帧取 joints / 2D 关键点），所以下游一行都不用改：
    高通、归一化、切点处理看到的还是一条普通序列。

    **相机字段故意留空**：各段的相机系不同，拼出来的东西没有统一的 K / cam_to_world
    可言。留空之后 `can_project` 会返回 False，`--de-bias` 自己就会跳过并说明原因
    （那里本来就有这条分支）——比硬凑一个假的相机矩阵诚实。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("人物时间线必须是非空数组")
    rng = []
    for it in raw:
        f, t = int(it["from"]), int(it["to"])
        src = Path(it["npz"]) if it.get("npz") else primary
        if t <= f:
            raise ValueError(f"区间非法：{f}~{t}")
        rng.append((f, t, src))
    rng.sort(key=lambda r: r[0])
    if rng[0][0] != 0:
        raise ValueError(f"必须从第 0 帧开始（现在是 {rng[0][0]}）")
    for a, b in zip(rng, rng[1:]):
        if b[0] != a[1]:
            raise ValueError(f"区间必须首尾相接，但 {a[1]} 与 {b[0]} 之间断了")

    cache: dict[Path, object] = {}

    def seq_of(p: Path):
        if p not in cache:
            cache[p] = sources.from_smplx_npz(str(p), fps=fps, with_mesh=False)
        return cache[p]

    parts = {k: [] for k in ("joints", "keypoints_2d", "keypoints_conf",
                             "keypoints_visible", "foot_contact")}
    first = seq_of(rng[0][2])
    for f, t, p in rng:
        sq = seq_of(p)
        for k in parts:
            v = getattr(sq, k, None)
            if v is None:
                parts[k] = None
            elif parts[k] is not None:
                parts[k].append(np.asarray(v)[f:t])
    stitched = MotionSequence(
        name=f"stitched:{Path(path).stem}", fps=first.fps,
        kp_to_joint=getattr(first, "kp_to_joint", None),
        image_size=getattr(first, "image_size", None),
        source=f"person-timeline:{path}",
        **{k: (np.concatenate(v, axis=0) if v else None) for k, v in parts.items()},
    )
    seams = [f for f, _, _ in rng[1:]]
    return stitched, seams


def slice_seq(seq, a: int, b: int):
    """取第 [a, b) 帧，返回一个新的 MotionSequence。

    相机那几项**也一起切**（它们是逐帧数组），这样段内仍然能做幅度去偏。
    """
    kw = {}
    for k in ("joints", "keypoints_2d", "keypoints_conf", "keypoints_visible",
              "foot_contact", "K", "cam_to_world"):
        v = getattr(seq, k, None)
        if v is not None:
            kw[k] = np.asarray(v)[a:b]
    return MotionSequence(
        name=f"{getattr(seq, 'name', 'seq')}[{a}:{b}]", fps=seq.fps,
        kp_to_joint=getattr(seq, "kp_to_joint", None),
        image_size=getattr(seq, "image_size", None),
        source=getattr(seq, "source", ""), **kw)


def parse_range(spec: str, n: int) -> tuple[int, int]:
    try:
        a_s, b_s = spec.split(":")
        a, b = int(a_s), int(b_s)
    except ValueError:
        raise ValueError(f"--range 要写成 A:B，收到 {spec!r}") from None
    a = max(0, a)
    b = n if b <= 0 else min(n, b)
    if b <= a:
        raise ValueError(f"--range {spec} 是空区间（素材共 {n} 帧）")
    return a, b


def can_project(seq) -> bool:
    """能否把关节投回像素去和 2D 关键点比。

    需要内参 K **和**外参 cam_to_world 同时存在。**global 版 npz 两个都没有**
    （它已经转到世界系，不需要相机），所以幅度保真度那几项在它上面算不了。
    以前这里没有检查，直接把 global 喂进来会在报告阶段抛
    `TypeError: 'NoneType' object is not subscriptable`——而产物早已写完，
    结果就是"文件是对的，但退出码 1"，任何自动化都会判成失败。
    """
    return seq.k_matrix(0) is not None and seq.world_to_cam(0) is not None


def write_funscript(path: Path, actions: list[dict]) -> None:
    path.write_text(json.dumps({"version": "1.0", "inverted": False,
                                "range": 100, "actions": actions},
                               ensure_ascii=False), encoding="utf-8")


def run(npz: Path, args: argparse.Namespace, seq=None) -> int:
    """主流程：由 seq 算出各轴、写文件、打印报告。

    seq 允许由调用方传入。Web 后端把它常驻在内存里，改一次参数只重跑这一半——
    读 npz 才是贵的那一半（尤其 vertices），算这一半只要几十毫秒。
    """
    seams: list[int] = []
    if seq is None and args.person_timeline:
        seq, seams = load_person_timeline(Path(args.person_timeline), Path(npz),
                                          fps=args.fps or 30.0)
    elif seq is None:
        # with_mesh=False：本脚本只用 joints / keypoints_2d / 相机，
        # vertices（10475 点 × T 帧）读进来要 1.7 秒却一次都用不到。
        seq = sources.from_smplx_npz(str(npz), fps=30.0, with_mesh=False)  # 内部会 to_world()

    # 只算一段：**必须排在序列加载之后**（切片要知道总帧数），也要排在时间线拼接
    # 之后（两者可以一起用：先按人物拼，再只看其中一段）。
    #
    # frame0 是这段在**整条素材**里的起始帧。写 funscript 的时间戳用它，产物就带
    # 绝对时间，几段可以直接按时间拼起来——这是"每段独立成篇"能拼的前提。
    frame0 = 0
    if args.range:
        a, b = parse_range(args.range, seq.n_frames)
        seq = slice_seq(seq, a, b)
        frame0 = a
        print(f"只算帧 {a}~{b}（{b - a} 帧，{b - a:.0f} 即为长度）。时间戳写绝对帧号。")

    J = np.asarray(seq.joints).astype(np.float64)
    n = seq.n_frames

    # ---- 帧率 ----
    fps = args.fps
    if fps is None and args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()
    if not fps or fps <= 0:
        fps = 30.0
        print(f"⚠ 未能读到帧率，按 {fps:g} 假设。用 --fps 显式指定更稳妥。", file=sys.stderr)

    # ---- 姿态自检 ----
    # **不要假设世界系的 Y 是"上"**：实测 GVHMR 的世界系 +Y 指向下方
    # （由 spine1-pelvis 导出的"上"与世界 +Y 的夹角余弦约 -0.97）。
    # 转换本身不依赖世界轴标签（基向量是从关节导出的），但自检必须用导出的"上"。
    _up = J[:, 3, :] - J[:, 0, :]
    _up /= np.maximum(np.linalg.norm(_up, axis=1, keepdims=True), 1e-9)
    up_mean = _up.mean(axis=0)
    rel = lambda i: float(((J[:, i, :] - J[:, 0, :]) * _up).sum(axis=1).mean())
    if not (rel(15) > 0.2 and rel(7) < -0.2):      # 头应在骨盆上方、脚在下方
        print(f"⚠ 身体姿态反常：头沿'上'投影 {rel(15):+.3f}，脚 {rel(7):+.3f}。"
              "世界系可能没有重力对齐，L0/R0 的物理意义就不成立。", file=sys.stderr)
    if up_mean[1] < 0:
        print(f"（提示：世界系 +Y 指向**下方**，与'上'的夹角余弦 {up_mean[1]:+.3f}。"
              "这是 GVHMR 的世界系约定，不影响转换——基向量取自关节而非世界轴。）")

    name = args.name or npz.parent.name
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 切点 ----
    # 只读清单，**不在这里检测**：检测要解码整条视频（几秒到几十秒），而这里
    # 是一次 0.25 秒量级的重算——把解码塞进来，界面上拖一次滑杆就要等十几秒。
    # 检测由 scene_cuts.py 单独跑，Web 那条路在打开素材时顺手做掉。
    cuts: list[int] = []
    if args.cuts:
        import scene_cuts
        cuts = scene_cuts.cut_frames(args.cuts, n)
        info = scene_cuts.load(args.cuts) or {}
        if info.get("n_frames") and abs(int(info["n_frames"]) - n) > 2:
            print(f"⚠ 切点清单是在 {info['n_frames']} 帧上测的，这份重建是 {n} 帧。"
                  "帧号可能对不上（同一条素材重跑过别的分辨率？），越界的已丢掉。",
                  file=sys.stderr)
        if cuts and not (args.cut_reset or args.cut_blank > 0):
            print("（给了切点但处理开关都是关的，输出不受影响）")
    # 只算一段时，**段的起点本身也是接缝**：段外的重建结果我们知道，但这里刻意
    # 不跨段看——每段独立成篇的意思就是"段外的数据不参与本段的滤波状态"。
    if args.range and args.cut_reset and frame0 > 0:
        cuts = sorted(set(cuts) | {0})
    # 人物切换的接缝也当成切点：两个人在各自世界系里重建，接缝处绝对位置不可比，
    # 和镜头切换是同一类不连续。列进 cuts 之后，去阶跃/掩蔽开着就自动生效。
    if seams:
        cuts = sorted(set(cuts) | set(s for s in seams if 0 < s < n))
        print(f"人物时间线：{len(seams) + 1} 段，接缝 {seams} 已并入切点")
    if args.cut_scale:
        print("⚠ --cut-scale 没有实现（见 --help 里的说明：单目下焦距误差不可观测），"
              "本次已忽略。", file=sys.stderr)

    # ---- 骨盆姿态 + 中立位 ----
    M = np.stack([orthonormal(m) for m in pelvis_frames(J)])          # (T,3,3)
    M0 = orthonormal(M.mean(axis=0)) if args.neutral == "mean" else M[0]
    R_loc = np.stack([M0.T @ m for m in M])                           # 相对中立位，表达在中立系
    eul_raw = Rotation.from_matrix(R_loc).as_euler(args.order, degrees=True)
    # 欧拉角天然会在 ±180 处跳变。骨盆俯仰实测出现 334° 的范围——那是环绕，不是真动作。
    # 逐列解环绕；不做的话设备会被要求做一个 300 多度的急转。
    eul = np.stack([np.rad2deg(np.unwrap(np.deg2rad(eul_raw[:, k])))
                    for k in range(3)], axis=1)
    wrapped = int(np.sum(np.abs(eul - eul_raw) > 1.0))
    if wrapped:
        print(f"（欧拉角解环绕：{wrapped}/{eul.size} 个分量被展开）")

    # 局部轴：x=右 y=上 z=前。scipy 的 as_euler(seq) 返回顺序**与 seq 字符串一致**，
    # 所以必须按字母查下标，不能写死 0/1/2（写死过一次，把扭胯错接到了 R2 上）。
    #   R0 = 绕上(竖直) = 扭胯        -> eul 里的 'Y' 位
    #   R1 = 绕前         = 翻滚      -> 'Z' 位
    #   R2 = 绕右         = 俯仰      -> 'X' 位
    #
    # **R2 不取负**。这里原来写的是 `-eul[:, iX]`（注释说"L2 的正方向是使用者
    # 左侧，所以绕左 = -绕右"），方向是**反的**：绕 +x（右）按右手定则把"上"转向
    # "前"，也就是骨盆向前倾；取负之后 R2 变大反而表示向后倾。2026-10-17 实测
    # （BV1RZY4zPEw2，世界系）：corr(R2_raw, 骨盆前倾角) = -0.66，R2 最大的 10%
    # 帧平均前倾 +4.4°，最小的 10% 帧 +19.2°——即"R2 越大越往后仰"。
    # 而设备侧的正方向是"前"（和 L2 正方向取使用者左侧是同一套约定），于是
    # 模拟器里看到的俯仰与人体相反。翻转之后：
    #   · 位移→旋转补偿（--comp-pitch）的默认负增益才真的是"反向抵消"
    #     （向前走了就向后倾）——之前它其实是在放大偏移；
    #   · 脚本的 R2 曲线整体关于 50 镜像，其余逻辑与符号无关。
    iY, iZ, iX = args.order.index("Y"), args.order.index("Z"), args.order.index("X")
    ang_abs = {"R0": eul[:, iY], "R1": eul[:, iZ], "R2": eul[:, iX]}
    blank_after = int(round(args.cut_blank * fps))
    # 报告要用到的**未处理**副本：报告里"不处理"那一列是拿原始信号重跑一次
    # 不带 cuts 的高通，所以留的是原始数组，不是先算好的数。
    raw0: dict[str, np.ndarray] = {k: v.copy() for k, v in ang_abs.items()}
    # 对朝向做高通：去掉累积的转向，留下扭胯的摆动。见 --highpass-rot 的说明。
    # 跨切点世界系的朝向一样会跳，所以旋转走同一套处理。
    hp_rot = args.highpass if args.highpass_rot is None else args.highpass_rot
    ang = {k: highpass(v, fps, hp_rot, cuts if args.cut_reset else None)
           for k, v in ang_abs.items()}
    # 掩蔽排在**高通之后**，理由见下面平移通道那段（差别在切点窗内的偏离，不在落差）
    for k in ang:
        blank_cuts(ang[k], cuts, blank_after)
    turn = ang_abs["R0"][-1] - ang_abs["R0"][0]

    # ---- 平移通道（基于中立位所在的骨盆位置）----
    #
    # 参考点。默认是骨盆关节本身；「阴部跟踪」取骨盆前下方的接触点，
    # 这时要做**杆臂修正**：
    #
    #   接触点是刚体上的一个固定点，不是世界系里的固定坐标。骨盆一转，
    #   它就绕着骨盆关节划弧。所以要拿逐帧的骨盆姿态矩阵 M（列 = [右, 上, 前]）
    #   把局部偏移旋到世界系：
    #
    #       p_接触(t) = p_骨盆(t) + M(t) · [0, -down, fwd]
    #
    #   少了这一项，骨盆一俯仰/侧倾，接触点的真实位移就完全跟不出来——
    #   杆臂越长、转动越快，差得越多。举个数：骨盆绕左右轴俯仰 20°、
    #   杆臂 0.12 m，接触点就横向走了约 4 cm，而骨盆本身没动。
    #
    # 姿态（R0/R1/R2）两种模式**共用**：接触点是同一刚体上的点，
    # 朝向就等于骨盆朝向。所以只有平移通道会变。
    P = J[:, 0, :]
    if args.reference == "genital":
        # 局部系分量顺序与 pelvis_frames 的列一致：[右, 上, 前]
        off_local = np.array([0.0, -abs(args.ref_down), abs(args.ref_fwd)])
        lever = np.einsum("tij,j->ti", M, off_local)      # (T,3)，M 的列是 [右,上,前]
        P = P + lever
    d = P - P.mean(axis=0) if args.neutral == "mean" else P - P[0]
    # 投影到中立位局部轴：上=R0轴、前=R1轴、左右=L2轴（左 = -右）
    up0, fwd0, right0 = M0[:, 1], M0[:, 2], M0[:, 0]

    # 报告用：未经处理的三个通道。**必须在 d 被改动之前取**，而且要在同样的
    # 位置做投影——报告里的"不处理"一列就是拿它重跑一次不带 cuts 的高通，
    # 顺序差一步就不等于旧行为了。
    raw0.update({"L0": d @ up0, "L1": d @ fwd0, "L2": -(d @ right0)})

    # 每镜头尺度归一（--cut-scale）**没有实现**，见该参数的说明：实测自动估计在这条
    # 管线上没有信号——体型先验把焦距误差吸收进了深度，输出里查不到。这里只挡一道。

    d = np.stack([highpass(d[:, k], fps, args.highpass,
                           cuts if args.cut_reset else None) for k in range(3)],
                 axis=1)
    tra = {"L0": d @ up0, "L1": d @ fwd0, "L2": -(d @ right0)}

    # 掩蔽排在**高通之后**。
    #
    # 两种顺序的"切点落差"都是 0（去阶跃已经把台阶吃掉了，掩蔽插在哪都改不了
    # 这件事），差别在**切点之后那一窗偏离了多少**：掩蔽写在高通之前的话，原始
    # 台阶被摊成一个 2W 秒的斜坡交给高通，而斜坡的低频部分照样漏得出去——
    # 实测本素材切点后 0.5s 的窗内偏离（L0/L1）：在后 39/119mm、
    # 在前 148/362mm（该素材真实动作的交流幅度是 5~15mm）。
    # 掩蔽在后时去阶跃先吃掉台阶，掩蔽只需把回正的那 0.3s 插直。
    for k in tra:
        blank_cuts(tra[k], cuts, blank_after)

    # ---- 归一化到 [0,1)，0.5 = 中立位 ----
    half = {"L0": args.travel_l0, "L1": args.travel_l1, "L2": args.travel_l2,
            "R0": args.max_deg, "R1": args.max_deg, "R2": args.max_deg}
    unit = {k: ("m" if k[0] == "L" else "deg") for k in AXES}

    # 幅度去偏：直接拿重建幅度比关键点幅度会系统性低估（回归稀释），
    # 用 k = 斜率/r² 修正后按 1/k 放大。见 estimate_retention 的推导。
    if args.de_bias:
        if can_project(seq):
            k_rot, k_tra = estimate_retention(seq)
            ang = {k: v / k_rot for k, v in ang.items()}
            tra = {k: v / k_tra for k, v in tra.items()}
            print(f"幅度去偏：旋转保留率 {k_rot:.2f}（放大 {1/k_rot:.2f}x），"
                  f"平移保留率 {k_tra:.2f}（放大 {1/k_tra:.2f}x）")
        else:
            print("⚠ --de-bias 需要把关节投回像素与关键点比较，"
                  "而这份 npz 没有相机内外参；已跳过（输出未做去偏）。")

    raw = {**tra, **ang}
    flips = {s.strip() for s in args.flip.split(",") if s.strip()}

    # 按时间窗归一化时，增益包络的平滑核长度（样本）。
    # 样本进出窗会让窗内分位出现台阶，增益跟着跳变——那是**动作里没有的**高频，
    # 真机会当成一次急停/急起。用窗长的 1/4 做零相位平滑（smooth 是居中的）。
    # 窗口本身已经很长，再平滑 1/4 不会把瞬态吃掉。
    env_k = max(3, int(round(args.fit_window * fps / 4)) | 1)
    # 窗长超过素材长度时 window_scale 会退化成全程（见那里的说明）。这在界面上
    # 是可能发生的——一条 8 秒的素材配 20 秒的窗。要说明，否则"我明明选了按时间窗，
    # 怎么和全程一模一样"。
    win_degenerate = 0 < n <= max(3, int(round(args.fit_window * fps)) | 1)
    win_on = args.fit_window > 0
    pct_win = args.fit if args.fit > 0 else WIN_PCT_FALLBACK

    norm, stats = {}, {}
    for k in AXES:
        v = raw[k].copy()
        if k in flips:
            v = -v
        v = smooth(v, args.smooth)
        span = float(v.max() - v.min())

        # ---- 阶段 1：全程缩放。按百分位把信号缩放到行程内——真实平台的行程是
        # 硬限制，正确做法是整体缩放（等比保留动作形状），而不是把超出的部分削平。
        # --fit 0 = 不做这一阶段，后面直接按 --travel-* 硬裁剪。
        fit_scale = 1.0
        if args.fit > 0:
            q_global = float(np.percentile(np.abs(v), args.fit))
            if q_global > 1e-9:
                fit_scale = half[k] / q_global
                v = v * fit_scale

        # ---- 阶段 2：按时间窗归一化。**与阶段 1 独立**：先换成"整个行程"为单位
        # （1.0 = 走到行程极限），half 在这里就约掉了，所以本阶段不需要知道行程是
        # 多少米/多少度，也不需要阶段 1 是否开过。
        d = v / half[k]
        gain_min = gain_max = 1.0
        capped = 0.0
        if win_on and not win_degenerate:
            d, gain_min, gain_max, capped = window_normalize(
                d, fps, args.fit_window, pct_win, env_k)

        u = 0.5 + 0.5 * d                            # 0.5 = 中立
        clipped = float(np.mean((u < 0) | (u > 1)))
        u = np.clip(u, 0.0001, 0.9999)
        norm[k] = u
        stats[k] = {"phys_min": float(v.min()), "phys_max": float(v.max()),
                    "phys_range": span, "unit": unit[k], "half": half[k],
                    "clipped": clipped, "fit_scale": fit_scale,
                    "gain_min": gain_min, "gain_max": gain_max, "capped": capped}

    # ---- 后处理（输出前）----
    # 放在归一化之后、时间轴重采样与写文件之前：这几步作用的是"设备最终会走的
    # 那个 0~1 刻度"，而不是物理量——换了归一化参数，高度限位的作用点也跟着变，
    # 这正是我们要的（它管的是机器的行程，不是人体的位移）。
    # 顺序由 --post-order 定（界面拖动排出来的），因为它们互不交换。
    post_stats: dict = {}
    args.post_order = parse_post_order(args.post_order)
    gains = {k: getattr(args, f"gain_{k.lower()}", 1.0) for k in GAIN_AXES}
    if args.height_limit or args.compensate or any(
            abs(float(v) - 1.0) > 1e-12 for v in gains.values()):
        norm, post_stats = postprocess(
            norm, args.post_order, args.height_limit, args.hl_start, args.hl_top,
            args.compensate, args.comp_pitch, args.comp_roll, gains)

    # ---- 时间轴 ----
    if args.rate and abs(args.rate - fps) > 1e-6:
        t_src = np.arange(n) / fps
        t_dst = np.arange(0, t_src[-1] + 1e-9, 1.0 / args.rate)
        norm = {k: np.interp(t_dst, t_src, norm[k]) for k in AXES}
        n = len(t_dst); fps_out = args.rate
        print(f"⚠ 已按 {args.rate:g}Hz 重采样（插值帧是合成的，不是重建出来的）")
    else:
        fps_out = fps

    # ---- 写 funscript ----
    # 主轴上单轴 1.0 文件（兼容性最好），次要轴按 MultiFunPlayer 的后缀约定分文件；
    # 另出一份 1.1 多轴合体文件。
    written = []
    axes_out = []
    for k in AXES:
        actions = [{"at": int(round((i + frame0) * 1000.0 / fps_out)),
                    "pos": int(round(float(p) * 100))} for i, p in enumerate(norm[k])]
        axes_out.append({"id": k, "actions": actions})
        if k == "L0":
            p = out / f"{name}.funscript"
        else:
            p = out / f"{name}.{SUFFIX[k]}.funscript"
        write_funscript(p, actions)
        written.append(p)
    multi = out / f"{name}.multi.funscript"
    multi.write_text(json.dumps({"version": "1.1", "inverted": False, "range": 100,
                                 "axes": axes_out}, ensure_ascii=False), encoding="utf-8")
    written.append(multi)

    # ---- TCode 指令流 ----
    # funscript 是"脚本"（由播放器随视频回放），而 osr-emu 的 write() 吃的是 TCode 指令。
    # 两者都要，否则没法直接喂给模拟器验证。
    #
    # 格式取自 TCode v0.3 规范的示例：字母 + **1 位轴号** + 值数字（位数即精度）。
    #   L277 -> 轴2, 值 0.77 ；L0000 -> 轴0, 值 0.000 ；L1000 -> 轴1, 值 0.000
    # 我们用 4 位值（0.9999 是规范的上限）。**这个位数约定请先用 osr-emu 确认**。
    iv = int(round(1000.0 / fps_out))
    lines = []
    for i in range(len(norm["L0"])):
        parts = []
        for k in AXES:
            v = int(round(float(norm[k][i]) * 9999))
            parts.append(f"{k}{v:04d}")
        parts.append(f"I{iv}")
        lines.append(" ".join(parts))
    tc = out / f"{name}.tcode.txt"
    tc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(tc)

    # ---- 报告 ----
    print(f"\n输入 {npz.name}   {n} 帧 @ {fps_out:g}fps   {n/fps_out:.1f}s")
    print(f"中立位 = {args.neutral}，欧拉顺序 = {args.order}，"
          f"高通 平移 {args.highpass:g}Hz / 旋转 {hp_rot:g}Hz，平滑 = {args.smooth} 点")
    # 后处理：**报出它实际做了什么**。这两项会改掉输出，不写清楚的话，用户只会
    # 看到"行程变小了/旋转跑偏了"，而不知道该去关哪一个开关。
    if post_stats.get("order"):
        print("后处理（输出前）顺序：" + " → ".join(
            POST_STEP_NAME[x] for x in post_stats["order"]))
    if "gain" in post_stats:
        g = post_stats["gain"]
        print("后处理 C 逐轴倍乘：" + "  ".join(
            f"{k}×{v['gain']:g}" + (f"(削顶 {v['clipped']:.1%})" if v["clipped"] > 0 else "")
            for k, v in g.items()))
    if "height_limit" in post_stats:
        h = post_stats["height_limit"]
        print(f"后处理 A 高度限位：≥{h['start']:g} 起收，顶端保留 {h['top']:g}×，"
              f"L1/L2 平均收到 {h['mean_g']:.3f}×（最紧 {h['min_g']:.3f}×）")
    for axis, c in (post_stats.get("compensate") or {}).items():
        print(f"后处理 B 位移补偿：{c['src']} → {axis}，增益 {c['gain']:g}，"
              f"该轴平均偏移 {c['mean_shift'] * 100:.1f} 点"
              + (f"，削顶 {c['clipped']:.1%}" if c["clipped"] > 0 else ""))

    # ---- 切点报告 ----
    # 关键的对照是"不做跨切处理会是什么样"：所以拿原始信号（raw0）**再跑一次
    # 不带 cuts 的高通**当基准。多跑六个长度为 n 的循环，几毫秒，换来的是一个
    # 真实的 before/after，而不是"台阶乘 0.9"这种估算。
    if cuts:
        print(f"切点 {len(cuts)} 个：" + "  ".join(
            f"帧{c}({c / fps:.2f}s)" for c in cuts))
        on = []
        if args.cut_reset:
            on.append("重置滤波状态")
        if args.cut_blank > 0:
            on.append(f"掩蔽 切点后 {args.cut_blank:g}s（{blank_after} 帧）")
        print(f"  处理：{'、'.join(on) if on else '无（开关都是关的）'}")
        print("  切点处**那一帧的落差**（各切点取最大，半行程为单位；"
              "1.00× = 一帧走完整个半行程）：")
        print(f"    {'轴':3s} {'原始台阶':>13s} {'= 半行程':>9s} "
              f"{'不处理':>8s} → {'已处理':>8s}")
        for k in AXES:
            steps = cut_steps(raw0[k], cuts)
            if not steps:
                continue
            is_l = k[0] == "L"
            fc = args.highpass if is_l else hp_rot
            before = cut_jump(highpass(raw0[k], fps, fc), cuts)
            after = cut_jump(tra[k] if is_l else ang[k], cuts)
            h = half[k]
            mx = max(steps, key=abs)
            print(f"    {k:3s} {mx:+9.4f} {'m' if is_l else '°'} "
                  f"{abs(mx) / h:8.2f}× {before / h:7.2f}× → {after / h:7.2f}×")
        if args.cut_blank > 0:
            print(f"  掩蔽吃掉 {len(cuts) * (blank_after + 1)} 帧"
                  f"（整段的 {len(cuts) * (blank_after + 1) / n:.1%}）——"
                  f"这是切点之后真实动作被换成直线的那部分，调小一点代价更低")
        print("  '不处理' = 拿原始信号重跑一次不带 --cuts 的高通。")
    # 归一化是两段，可以各自开关。--fit-window 是**独立的第二段**，不依赖 --fit。
    stage1 = (f"全程：|值的 {args.fit:g}% 分位 → 行程极限" if args.fit > 0
              else "全程：关（不缩放，直接按行程硬裁剪）")
    if win_on and win_degenerate:
        print(f"归一化 = {stage1}")
        print(f"  注：窗长 {args.fit_window:g}s 已超过素材长度 {n / fps:.1f}s，"
              f"每个窗都装着整段，所以退化成全程——结果与不开窗一致。")
    elif win_on:
        print(f"归一化 = {stage1}")
        print(f"         + 时间窗：窗长 {args.fit_window:g}s（±{args.fit_window / 2:g}s），"
              f"窗内 |值的 {pct_win:g}% 分位 → 行程极限"
              + (f"（--fit 关着，取惯例分位 {WIN_PCT_FALLBACK:g}%）" if args.fit <= 0 else ""))
        print(f"         增益封顶 {MAX_BOOST:g}x（相对全程），包络平滑 {env_k} 点")
        print(f"         第二段开启时 --travel-* / --max-deg 不影响输出"
              f"（半行程在'整个行程'为单位里已经约掉了）")
    else:
        print(f"归一化 = {stage1}")
    if abs(turn) > 90:
        print(f"⚠ 骨盆偏航在整段里累积转了 {turn:+.0f}°（近 {abs(turn)/360:.1f} 圈）。"
              f"设备跟不了这个；旋转高通已把它去掉，只保留扭胯的摆动了。"
              f"若要保留转向请设 --highpass-rot 0 并自行确认行程够不够")
    if args.flip:
        print(f"已翻转符号: {args.flip}")
    win_used = win_on and not win_degenerate
    print(f"\n{'轴':4s} {'含义':24s} {'实测范围':>13s} "
          f"{'窗口增益' if win_used else '自动适配':>16s} {'裁剪':>7s}")
    for k in AXES:
        s = stats[k]
        if win_used:
            # 逐帧增益，所以报范围；被下限托住的帧单独标出来
            fit = f"{s['gain_min']:7.2f}-{s['gain_max']:.2f}x"
            if s["capped"] > 0.001:
                fit = f"{s['gain_min']:6.2f}-{s['gain_max']:.2f}x*"
        elif args.fit <= 0:
            fit = "            关闭"
        else:
            fit = f"{s['fit_scale']:9.3f}x"
        print(f"  {k:3s} {AXIS_DESC[k]:24s} {s['phys_range']:10.3f}{s['unit']:4s}"
              f"{fit} {s['clipped']:6.1%}")
    if win_used:
        cap_max = max(stats[k]["capped"] for k in AXES)
        print(f"  实测范围 = 人实际动的幅度；窗口增益 = 逐帧缩放系数（最小-最大），"
              f"1.00x 表示该帧与全程归一化给出的尺度相同。")
        # 只有真触到封顶才说"靠封顶避免了噪声放大"——否则这句话在描述一件没发生的事
        if cap_max > 0.001:
            print(f"  **带 * 的轴有帧触到了增益上限 {MAX_BOOST:g}x**"
                  f"（最高 {cap_max:.1%} 的帧）——那些段落人几乎没动，靠封顶避免了"
                  f"把跟踪噪声放大成满行程。")
        else:
            print(f"  没有帧触到增益上限 {MAX_BOOST:g}x（各段动作幅度都够大），"
                  f"所以表里没有 * 标记。")
        print(f"  ⚠ 按时间窗归一化**改写了动作的相对幅度**：安静段落被抬起来，"
              f"设备做的事比人大。要保留真实幅度关系请把「归一化窗口」调回全程。")
    elif args.fit > 0:
        print(f"  实测范围 = 人实际动的幅度；自动适配 = 为塞进行程做的等比缩放"
              f"（按 |值| 的 {args.fit:g}% 分位映射到半步程，"
              f"L 轴 {args.travel_l0:g}m / R 轴 {args.max_deg:g}°）")

    if can_project(seq):
        rep = amplitude_report(seq)
        print(f"\n幅度保真度（重建 vs 2D 观测）")
        print(f"  扭胯(R0) 原始比值 {rep['gain_twist']:.2f}（相关 {rep['r_twist']:+.2f}）"
              f"   平移 原始比值 {rep['gain_trans']:.2f}")
        kr, kt = estimate_retention(seq)
        print(f"  去偏后保留率：旋转 {kr:.2f}（丢失 {100*(1-kr):.0f}%）  "
              f"平移 {kt:.2f}（丢失 {100*(1-kt):.0f}%）   <- 这个才是可信的幅度比")
        print(f"  原始比值被回归稀释压低（关键点带噪声使 OLS 斜率偏小），别直接当幅度比")
        if args.de_bias:
            print(f"  已按 1/k 补齐：旋转 {1/kr:.2f}x，平移 {1/kt:.2f}x")
        else:
            print(f"  加 --de-bias 可按 1/k 补齐幅度")
    else:
        print(f"\n幅度保真度：**跳过**")
        print(f"  这项诊断要把重建的关节投回像素、与 2D 关键点逐帧比较，需要相机内外参；")
        print(f"  而 global 版 npz 已经转到世界系，不带 K / cam_to_world。")
        print(f"  需要这项诊断请改用 incam 版。")
    print(f"\n已写出 {len(written)} 个文件到 {out}")
    for p in written:
        print(f"  {p.name}")
    print("\n下一步：用 osr-emu 验证方向与幅度（别直接上硬件）：")
    print(f"  new OSREmulator('#canvas', {{model:'SR6'}}); 再逐行 write() 里面的 TCode")
    # --fit 99 本身就是"允许 1% 削顶"的意思，所以阈值要高于它，否则每次都误报
    thr = 0.03 if args.fit > 0 else 0.005
    if any(stats[k]["clipped"] > thr for k in AXES):
        print(f"\n⚠ 有轴裁剪超过 {thr:.1%}（见上表）——该轴动作超出设定的半步程。"
              "调大 --travel-* / --max-deg，或调低 --fit 分位。")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数、校验输入，再交给 run()。

    Web 后端不走这里，它直接调 run() 并传入缓存好的 seq。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    args = parse_args(argv)
    npz = Path(args.npz).resolve()
    if not npz.exists():
        print(f"找不到 {npz}", file=sys.stderr)
        return 1
    return run(npz, args)


if __name__ == "__main__":
    raise SystemExit(main())
