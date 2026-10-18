"""网格从结果 npz 里拆出去之后，读写两端还对不对得上。

    python src/tests/test_mesh_split.py

背景：vertices 一个就占结果文件的 **99.5%**（10475 顶点 × 3 × float32 = 每帧
123 KB），而只有渲染和 viz 读它——3D 视图、算脚本、前端的 motion 接口都只用
joints。所以它单独落盘成 `x_incam.mesh.npz`，平时可以删掉，要渲染时补回来。

这里盯五件事：
  1. 命名规则（唯一一份，见 mesh_npz.py）
  2. 读取端能从**旁边那个文件**读到网格
  3. 网格文件不在时**不报错**——退化成"没有 mesh"，而不是把整条读取打断
     （`motion` 和算脚本走的就是这条路，它们本来就不读网格）
  4. 摘网格的脚本：摘完关节还在，且**同目录没有 .pt 的一条都不动**
     （网格由 .pt 重建，它不在就永远补不回来）
  5. 脚本别去处理自己的临时文件和 .mesh.npz —— 踩过：glob 的 `*.npz` 把
     `x.npz.tmp.npz` 也捞进来当成结果文件处理
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import strip_mesh  # noqa: E402
from mesh_npz import mesh_path_for  # noqa: E402
from server import paths  # noqa: E402
from viz.sources import from_smplx_npz  # noqa: E402

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def make_npz(path: Path, n: int = 4, with_mesh: bool = True) -> None:
    """造一份最小但形状合法的结果（+ 可选的网格），和真实产物同一个布局。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joints": np.zeros((n, 24, 3), dtype=np.float32),
        "K": np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
                      dtype=np.float32),
        "cam_to_world": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        "keypoints_2d": np.zeros((n, 17, 2), dtype=np.float32),
        "keypoints_conf": np.ones((n, 17), dtype=np.float32),
        "kp_to_joint": np.zeros(17, dtype=np.int32),
        "image_size": np.array([640, 480], dtype=np.int32),
        "fps": np.array([30.0], dtype=np.float32),
    }
    if with_mesh:
        payload["vertices"] = np.zeros((n, 10475, 3), dtype=np.float32)
        payload["faces"] = np.zeros((3, 3), dtype=np.int32)
    np.savez_compressed(path, **payload)


def main() -> int:
    print("\n[1] 命名规则")
    check(str(mesh_path_for("a/x_incam.npz")).endswith("x_incam.mesh.npz"),
          "x_incam.npz -> x_incam.mesh.npz")
    check(mesh_path_for("y_global.npz").name == "y_global.mesh.npz", "另一个坐标系同理")
    check(mesh_path_for("z").name == "z.mesh.npz", "没有扩展名也能用")

    tmp = Path(tempfile.mkdtemp(prefix="meshsplit_"))
    try:
        _body(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"✗ {len(FAILS)} 项没过：")
        for f in FAILS:
            print("   -", f)
        return 1
    print("✓ 全部通过")
    return 0


def _body(tmp: Path) -> None:
    print("\n[2] 读取端：网格在旁边那个文件里，能读出来")
    npz = tmp / "a" / "hmr4d_results_incam.npz"
    make_npz(npz, with_mesh=False)
    make_npz(mesh_path_for(npz), with_mesh=True)
    # 网格文件里也有 joints；读取端要优先用主 npz 的，别被覆盖
    with np.load(npz) as z:
        z_joints = z["joints"].copy()
    with np.load(mesh_path_for(npz)) as z:
        assert z["vertices"].shape[1] == 10475
    seq = from_smplx_npz(str(npz), with_mesh=True, fps=30.0)
    check(seq.vertices is not None and len(seq.vertices) == 4, "读到了 4 帧网格")
    check(seq.faces is not None, "读到了 faces")
    check(np.array_equal(np.asarray(seq.joints), z_joints), "joints 用的是主 npz 那份")

    print("\n[3] 网格文件不在：退化，不打断")
    npz2 = tmp / "b" / "hmr4d_results_incam.npz"
    make_npz(npz2, with_mesh=False)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seq2 = from_smplx_npz(str(npz2), with_mesh=True, fps=30.0)
        check(seq2.vertices is None, "没有网格就是 None，没抛异常")
        check(any("网格" in str(x.message) for x in w), "而且明确警告了一句")
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        from_smplx_npz(str(npz2), with_mesh=False, fps=30.0)
        check(not any("网格" in str(x.message) for x in w2),
              "with_mesh=False 时不该有这个警告（它本来就不要网格）")

    print("\n[4][5] 摘网格的脚本")
    out = tmp / "outputs"
    real = out / "SLUG1" / "gvhmr_rtmpose" / "V_1280p"
    make_npz(real / "hmr4d_results_incam.npz", with_mesh=True)
    (real / "hmr4d_results.pt").write_bytes(b"x")          # 有 .pt → 可以摘
    orphan = out / "SLUG2" / "gvhmr_rtmpose" / "V_1280p"
    make_npz(orphan / "hmr4d_results_incam.npz", with_mesh=True)   # 没有 .pt
    # 老版本留下的残件形态（那次 bug 的产物）。**别用 `.tmp.npz`** ——那是脚本
    # 自己的 scratch 名，会被正常地写掉，测不出"过滤没生效"这件事。
    stray = real / "hmr4d_results_incam.npz.strip-tmp.npz"
    make_npz(stray, with_mesh=True)
    already = out / "SLUG3" / "gvhmr_rtmpose" / "V_1280p"
    make_npz(already / "hmr4d_results_incam.npz", with_mesh=False)
    make_npz(mesh_path_for(already / "hmr4d_results_incam.npz"), with_mesh=True)
    (already / "hmr4d_results.pt").write_bytes(b"x")       # 已经是新格式

    real_out = paths.OUTPUTS
    paths.OUTPUTS = out
    argv = sys.argv
    sys.argv = ["strip_mesh.py", "--apply"]
    try:
        strip_mesh.main()
    finally:
        paths.OUTPUTS = real_out
        sys.argv = argv

    with np.load(real / "hmr4d_results_incam.npz") as z:
        keys = set(z.files)
        check("vertices" not in keys and "faces" not in keys, "摘掉的：vertices/faces")
        check("joints" in keys and z["joints"].shape == (4, 24, 3), "留住: joints 形状没变")
        check("K" in keys and "keypoints_2d" in keys, "留住: K / 2D 关键点")
    with np.load(orphan / "hmr4d_results_incam.npz") as z:
        check("vertices" in z.files, "**没有 .pt 的一条原样不动**（网格补不回来）")
    check(stray.exists(), "临时残件没被当成结果文件处理")
    with np.load(already / "hmr4d_results_incam.npz") as z:
        check("vertices" not in z.files, "已经是新格式的没被再动一遍")


if __name__ == "__main__":
    raise SystemExit(main())
