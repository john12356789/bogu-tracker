"""A/B：HMR2 特征用 fp16 前向，到底差多少。

    E:\\anaconda3\\envs\\gvhmr\\python.exe src/_fp16_ab.py <素材 slug>

**只改一个变量**：沿用现有的 `bbx.pt`（不重跑跟踪）和 `vitpose.pt`，只把
HMR2 特征分别用 fp32 / fp16 算一遍，然后各自走完 GVHMR 推理 + 导出，
比最后那一步的**关节**（以及 3D 顶点）差多少。

为什么要比到关节：特征只是中间量，差 1e-3 的特征如果在关节上放大成 5 cm，
那这个加速就不能要。反过来特征差得再多、关节在亚毫米级，那它就是安全的。

下划线开头：一次性探针，不是测试。
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GVHMR = ROOT / "GVHMR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "shims"))
sys.path.insert(0, str(GVHMR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.chdir(GVHMR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hmr4d.utils.preproc import Extractor  # noqa: E402
from prep_boxes import _extract_streaming  # noqa: E402

slug = sys.argv[1]
out = ROOT / "workspace" / "outputs" / slug
main_rt = next(d for d in sorted(out.glob("gvhmr_rtmpose*")) if "__t" not in d.name)
rt_dir = main_rt / next(d.name for d in main_rt.iterdir() if d.is_dir())
proc = next((ROOT / "workspace" / "processed").glob(f"{slug}_*.mp4"))
res_dir = rt_dir          # npz 就在这一层（rt_dir 本身就是 <proc_stem> 目录）
print(f"素材 {slug}\n  rt_dir {rt_dir}\n  结果   {res_dir}")

bbx = torch.load(rt_dir / "preprocess" / "bbx.pt", map_location="cpu")
bbx_xys = bbx["bbx_xys"].float()
print(f"  bbx_xys {tuple(bbx_xys.shape)}")

tmp = ROOT / "workspace" / "outputs" / "_fp16ab"
# GVHMR 按 <output_root>/<视频名>/preprocess/ 找那三个文件，所以要放深一层
sub = {t: tmp / t / proc.stem / "preprocess" for t in ("incam", "fp16")}
for d in sub.values():
    d.mkdir(parents=True, exist_ok=True)

# 两次都从同一块 bbx 出发，只有 half 不同
feats = {}
for tag, half in (("incam", False), ("fp16", True)):
    feat_file = sub[tag] / "vit_features.pt"
    if feat_file.exists():                     # 重跑时复用，不白算 5 分钟
        f = torch.load(feat_file, map_location="cpu")
        print(f"[{tag:>5}] 复用已有特征 {tuple(f.shape)}", flush=True)
        feats[tag] = f
        continue
    extractor = Extractor()
    if tag == "fp16":
        torch.cuda.empty_cache()
    t = time.time()
    f = _extract_streaming(extractor, str(proc), bbx_xys, 0.5, half=half)
    dt = time.time() - t
    del extractor
    torch.cuda.empty_cache()
    feats[tag] = f
    torch.save(f, feat_file)
    if tag == "incam":                # 对照组必须和现网那份**逐字节**一致，
        stored = torch.load(rt_dir / "preprocess" / "vit_features.pt", map_location="cpu")
        same = stored.shape == f.shape and torch.equal(stored, f)
        print(f"        重算的 fp32 特征与现网那份逐元素相同: {same}"
              f"{'' if same else '  ← 不同的话下面就得把它当噪声底看'}", flush=True)
    print(f"[{tag:>5}] {dt:6.1f}s  {dt / len(f) * 1000:5.1f} ms/帧   "
          f"形状 {tuple(f.shape)}  dtype {f.dtype}", flush=True)

a, b = feats["incam"].numpy(), feats["fp16"].numpy()
d = np.abs(a - b)
scale = max(float(np.abs(a).mean()), 1e-9)
print(f"\n=== 特征本身 ===")
print(f"  逐元素 |差|: 均值 {d.mean():.3e}  中位 {np.median(d):.3e}  最大 {d.max():.3e}")
print(f"  相对量级（对 |fp32| 的均值）: {d.mean() / scale:.3e}")

# ---- 各自跑完 GVHMR 推理 + 导出 ----
src_files = rt_dir / "preprocess" / "bbx.pt", rt_dir / "preprocess" / "vitpose.pt"
for tag in ("incam", "fp16"):
    d0 = tmp / tag
    for f in src_files:
        shutil.copy2(f, sub[tag] / f.name)
    t = time.time()
    p = subprocess.run([r"E:\anaconda3\envs\gvhmr\python.exe", str(HERE / "run_gvhmr.py"),
                        "--video", str(proc), "-s", "--output_root", str(d0)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    print(f"\n[{tag}] GVHMR 退出码 {p.returncode}  {time.time() - t:.0f}s", flush=True)
    if p.returncode:
        print((p.stdout or "")[-800:], (p.stderr or "")[-500:])
        raise SystemExit(1)
    # 导出到同一目录的另一种名字，免得和上面那个 vit_features.pt 混
    subprocess.run([r"E:\anaconda3\envs\gvhmr\python.exe", str(HERE / "export_gvhmr.py"),
                    str(next(d0.glob("**/hmr4d_results.pt"))), "--spaces", "incam"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))

# ---- 比关节 / 顶点 ----
ref = np.load(res_dir / "hmr4d_results_incam.npz")          # 现网产物（fp32 特征）
ref_mesh = None
mp = res_dir / "hmr4d_results_incam.mesh.npz"
if mp.exists():
    ref_mesh = np.load(mp, mmap_mode="r")["vertices"]

print("\n=== 重建结果对比（都相对现网产物）===")
print("  incam = fp32 重算（**对照组**：用的特征和现网那份一模一样，所以它俩的差")
print("          就是 GVHMR 自己的跑批不确定性，不是 fp16 造成的）")
for tag in ("incam", "fp16"):
    nz = next(d for d in (tmp / tag).rglob("hmr4d_results_incam.npz"))
    z = np.load(nz)
    ja, jb = ref["joints"], z["joints"]
    n = min(len(ja), len(jb))
    dj = np.abs(ja[:n] - jb[:n]) * 1000.0                   # 米 -> 毫米
    msg = (f"  {tag:>5}: joints 最大 {dj.max():8.3f} mm   逐元素均值 {dj.mean():7.4f} mm")
    mf = next((x for x in (tmp / tag).rglob("*.mesh.npz")), None)
    if ref_mesh is not None and mf is not None:
        va = np.asarray(ref_mesh[:n], dtype=np.float32)
        vb = np.load(mf, mmap_mode="r")["vertices"][:n]
        dv = np.abs(va - vb) * 1000.0
        msg += f"   顶点最大 {dv.max():8.3f} mm"
    print(msg, flush=True)

print("\n（对照组的数字就是噪声底。fp16 那行如果不是明显大于它，说明差别在噪声里。）")
