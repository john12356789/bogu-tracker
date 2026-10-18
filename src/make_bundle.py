"""把「数据目录」打成压缩包——桌面壳解压它就能跑，**不需要装任何东西**。

    python src/make_bundle.py                       # 打到 dist/骨骼追踪-数据.zip
    python src/make_bundle.py --dry                 # 只列清单和体积，不写文件
    python src/make_bundle.py --out D:/x.zip
    python src/make_bundle.py --fast                # 压缩级别 1
    python src/make_bundle.py --pipeline-env <路径> # 换一份 Python 环境
    python src/make_bundle.py --verify <zip>        # 检查一个已经打好的包

## 这个包和 exe 的分工

exe（electron-builder 出的 portable 单文件）里只有 Electron + 前端 dist，约 81 MB。
**所有重量都在这里**：模型权重 + 一个完整的 Python 环境。分开的理由是 portable
exe 每次启动都把自己解到临时目录——把 9 GB 放进去等于每次开软件解 9 GB。

## 为什么是**一个** Python 环境，不是两个

开发机上跑着两个（3.12 跑 Flask、3.10 跑管线）。但实测 2026-10-17：把 flask 和
flask-cors（合计 5 MB）补进管线那个环境之后，**它跑后端的接口一模一样**——
`/api/health`、`/api/clips`、`/api/params` 全部正常。所以包里只带一个。

**不能带 venv。** `python -m venv` 出来的东西不是自包含的：`pyvenv.cfg` 里写着
`home = C:\\Python312`，标准库是从基座安装加载的，拷到没有那个基座的机器上直接
起不来。所以拷的是 **conda 环境**（自带完整标准库），实测拷到别的路径后
`torch.cuda.is_available()` 为真、2048×2048 的 CUDA matmul 算得通。

flask 不装进那份环境里（`--target` 到 `python-extra/`），因为它只给后端用：
启动器把 `python-extra` 挂到 PYTHONPATH 上，子进程自然继承。这样源环境保持原样，
打包过程**不修改任何已有环境**。

## 为什么清单是**写死的**，不是"整棵树减去排除项"

这个包的内容是"哪些重量必须跟着走"的**决定**，应该一眼看得见。写成通配符加
排除规则的话，项目里新加一个目录就会静默漏掉（用户拿到包，跑到第 4 步才报
"找不到 xxx"），或者静默带上 10 GB。要什么逐条列，不要什么也逐条列。

## 省掉的东西是怎么定的

每一条都有据，不是猜的：

  · `body_models/{smpl,smplx}/*_{FEMALE,MALE}.*`  680 MB
      用审计钩子盯着文件打开实测：管线那三个加载点（make_smplx('supermotion')、
      SmplxLite、SmplLite）**只打开 NEUTRAL 那两个**。
  · `checkpoints/dpvo/dpvo.pth`  14 MB
      管线调 run_gvhmr.py 时带 `-s`（静态相机），load_data_dict 直接走
      `torch.eye(3)`，SLAM 整段被跳过。
  · `GVHMR/outputs/`  342 MB、`workspace/models/smplx/`  312 MB（只有 run_viz
      --demo 用）、`GVHMR/.git`  9 MB、`ffplay.exe`  98 MB。
  · 用户自己的数据（videos/processed/outputs/funscripts/web_cache，24 GB）——
    只建空目录，让后端一进去就有地方写。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 输出编码：**只在被重定向的时候**才显式改成 UTF-8。
#
# 重定向到管道/文件时，Python 用的是本地编码——在这台中文 Windows 上是 cp936
# （GBK）。GBK 编不出来的字符（拉美字母、emoji、某些符号）会让 print 直接抛
# UnicodeEncodeError，而这个脚本的输出是给用户看的日志，不该因为一个字符整个挂掉。
# 所以重定向时改成 UTF-8，并且 errors="replace" 兜底。
#
# 真实控制台不动它：Python 3.6+ 在 Windows 上对控制台本来就有自己的处理
# （PEP 528），双击 bat 时 stdout 是控制台（isatty() 为真），交给 Python 自己。
# ——控制台那条路我没有单独验过（这里没法挂一个真 tty），但改动只是"少做一件事"，
#    不会比原来更差。
for _s in (sys.stdout, sys.stderr):
    try:
        if not _s.isatty():
            _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

DEFAULT_OUT = ROOT / "dist" / "骨骼追踪-数据.zip"

# ---------------------------------------------------------------------------
# 清单
# ---------------------------------------------------------------------------
#: 要打进去的目录树（相对项目根）。
TREES = [
    "src",                    # 后端 + 管线脚本（1.7 MB）
    "GVHMR",                  # GVHMR 代码树（约 25 MB，inputs/outputs 另算）
    "workspace/models",       # RTMPose 等权重（595 MB 里挑 282 MB）
]

#: 要打进去的**单个文件**。GVHMR/inputs 整棵是跳过的（见 SKIP_TREES），
#: 只挑这几个必须的权重进来——顺序和 checks.cjs 里那张表一致。
FILES = [
    "GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    "GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    "GVHMR/inputs/checkpoints/yolo/yolov8x.pt",
    "GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    "GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
]

#: 要打进去的空目录：后端一进去就要有地方写。
EMPTY_DIRS = [
    "workspace/videos", "workspace/processed", "workspace/outputs",
    "workspace/funscripts", "workspace/web_cache", "workspace/presets",
    "workspace/logs", "workspace/emu",
]

#: **项目之外**要打进去的目录树：(源目录, 包内前缀, 说明)。
#: 只从 --pipeline-env 来，不做成可改的常量列表——它连同一条 pip 一起构成
#: "那个环境"，分开写容易对不上。
PIPELINE_ENV_DEFAULT = Path("E:/anaconda3/envs/gvhmr")

#: 补进包里的小依赖。**装到 python-extra/，不装进那份环境**——源环境保持原样，
#: 运行时由启动器挂到 PYTHONPATH 上。管线那个环境自己缺 flask。
#:
#: `pyserial` 是「连接设备」（把脚本发给真硬件）唯一的依赖。**它进了包**是有意的：
#: 那是唯一一个"新机器上不会自己长出来"的东西——管线环境里没有它（实测），
#: 而缺了它的表现是**串口下拉框永远空着**（`device.list_ports` 读不出 pyserial
#: 时返回空表而不是抛异常），用户看到的是"设备插着但列不出来"，
#: 离"少装了一个 Python 包"很远。它只有 ~200 KB。
EXTRA_PIP = ["flask", "flask-cors", "pyserial"]
EXTRA_PIP_DIR = "python-extra"

#: 从 PATH 上找、按名字挑单个文件的外部工具。(命令名, 包内路径)
TOOLS = [
    ("ffmpeg", "tools/ffmpeg/bin/ffmpeg.exe"),
    ("ffprobe", "tools/ffmpeg/bin/ffprobe.exe"),
    # ffplay 不打：98 MB，播放器，这个项目一次都不用
]

#: 按**路径前缀**排除（命中整棵子树）。
SKIP_TREES = [
    "GVHMR/inputs",              # 只有 checkpoints；要的那 5 个走 FILES
    "GVHMR/outputs",             # 342 MB 示例产物
    "GVHMR/.git",                # 9.4 MB 克隆元数据
    "GVHMR/third-party",         # 空子模块
    "workspace/models/smplx",    # 312 MB，只有 run_viz --demo 用
    "workspace/models/vposer_v1_0",   # 2.7 MB，没人引用
    "workspace/models/models",   # 空目录，历史残留
]

#: 按**目录名**排除（出现在任何层级都跳过）。环境里的 __pycache__ 尤其多，
#: 而且换机器之后一律失效，带上只是白占地方。
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git", ".idea"}

#: 已经是压缩格式的，再压一遍只是白烧 CPU。
STORE_EXT = {".npz", ".zip", ".gz", ".bz2", ".xz", ".7z", ".jpg", ".png", ".mp4",
             ".whl", ".tar", ".conda"}

COMPRESS = {False: 6, True: 1}


def _skipped(rel: str) -> bool:
    return any(rel == s or rel.startswith(s + "/") for s in SKIP_TREES)


def manifest(pipeline_env: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """返回 (要打的 (绝对路径, 包内路径), 警告)。"""
    out: list[tuple[Path, str]] = []
    warn: list[str] = []

    def walk(base: Path, prefix: str = ""):
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = (f"{prefix}/{p.relative_to(base).as_posix()}" if prefix
                   else p.relative_to(ROOT).as_posix())
            if _skipped(rel) or (SKIP_NAMES & set(p.parts)):
                continue
            out.append((p, rel))

    for tree in TREES:
        base = ROOT / tree
        if not base.exists():
            warn.append(f"清单里的 {tree} 不存在——这一项会整个漏掉")
            continue
        walk(base)

    for f in FILES:
        p = ROOT / f
        if not p.is_file():
            warn.append(f"**必需权重缺失：{f}**")
            continue
        out.append((p, f))

    for src, prefix, why in ((pipeline_env, "python", "解释器环境（后端与管线共用）"),):
        if not src.exists():
            warn.append(f"**{why}缺失：{src}**（--pipeline-env 可以换一个）")
            continue
        walk(src, prefix)

    for cmd, dest in TOOLS:
        found = shutil.which(cmd)
        if not found:
            warn.append(f"**PATH 上找不到 {cmd}**——它要打进包里，找不到就没得打")
            continue
        out.append((Path(found), dest))

    seen: set[str] = set()
    uniq = []
    for p, rel in out:
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append((p, rel))
    return uniq, warn


def pip_extras(env: Path, dest: Path) -> int:
    """把 flask 那几包装到 `dest`（独立目录，不进环境）。

    返回装进去的字节数。用 `--target` 而不是装进环境：**打包过程不修改任何
    已有环境**——那是一台机器上正在用的东西，被构建脚本改掉是最不该发生的事。
    """
    py = env / "python.exe"
    if not py.exists():
        print(f"  ⚠ {py} 不在，跳过 python-extra", file=sys.stderr)
        return 0
    cmd = [str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check",
           "--target", str(dest), *EXTRA_PIP]
    print(f"  装 {EXTRA_PIP} 到 python-extra/ …", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ⚠ pip 失败：{(r.stderr or r.stdout)[-400:]}", file=sys.stderr)
        return 0
    return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())


def _existing_ancestor(p: Path) -> Path:
    """往上找到第一个存在的目录。`disk_usage` 对不存在的路径会抛——
    而 --out 指定的目录在第一次打包时往往还不存在。"""
    p = p if p.is_absolute() else (Path.cwd() / p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def human(n: float) -> str:
    return f"{n / 1073741824:.2f} GB" if n >= 1073741824 else f"{n / 1048576:.0f} MB"


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------
def write(items: list[tuple[Path, str]], out: Path, fast: bool) -> None:
    level = COMPRESS[fast]
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(p.stat().st_size for p, _ in items)
    done = 0
    t0 = time.time()

    tmp = out.with_suffix(out.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=level,
                         allowZip64=True) as z:
        for i, (p, rel) in enumerate(items, 1):
            size = p.stat().st_size
            if p.suffix.lower() in STORE_EXT:
                zi = zipfile.ZipInfo.from_file(p, rel)
                zi.compress_type = zipfile.ZIP_STORED
                with open(p, "rb") as fh, z.open(zi, "w") as o:
                    while chunk := fh.read(1 << 20):
                        o.write(chunk)
            else:
                z.write(p, rel)
            done += size
            if size >= 200 << 20 or i % 5000 == 0:
                print(f"  {human(done)} / {human(total)}   {rel}", flush=True)
        for d in EMPTY_DIRS:
            if not any(rel.startswith(d + "/") for _, rel in items):
                z.writestr(zipfile.ZipInfo(d + "/"), b"")

    tmp.replace(out)
    print(f"\n写好：{out}")
    print(f"  {human(total)} -> 压缩后 {human(out.stat().st_size)}"
          f"（压掉 {(1 - out.stat().st_size / max(1, total)) * 100:.0f}%）")
    print(f"  {len(items)} 个文件，{time.time() - t0:.0f} 秒")


#: 整棵不该出现的路径。
FORBID_ANY = ("GVHMR/outputs/", "GVHMR/.git/", "workspace/models/smplx/",
              "workspace/models/vposer_v1_0/", "dpvo/", "__pycache__", ".pytest_cache/")
FORBID_NAME = ("SMPL_FEMALE", "SMPL_MALE", "SMPLX_FEMALE", "SMPLX_MALE", "ffplay")
#: 用户数据目录：空的目录条目本身是对的，底下多出任何东西都是错的。
FORBID_CONTENT = ("workspace/videos/", "workspace/processed/", "workspace/outputs/",
                  "workspace/funscripts/", "workspace/web_cache/")


def verify(out: Path) -> int:
    """检查一个包：该在的在不在，不该在的有没有混进去。"""
    bad = 0
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        print(f"{out}：{len(names)} 个条目，{human(out.stat().st_size)}")

        need = [
            ("src/serve.py", "后端入口"),
            ("GVHMR/hmr4d/__init__.py", "GVHMR 代码"),
            ("python/python.exe", "解释器环境"),
            ("python-extra/flask/__init__.py", "补进去的 flask"),
            ("tools/ffmpeg/bin/ffmpeg.exe", "ffmpeg"),
            ("tools/ffmpeg/bin/ffprobe.exe", "ffprobe"),
            *([(f, "必需权重") for f in FILES]),
        ]
        for f, why in need:
            ok = f in names
            print(f"  [{'PASS' if ok else 'FAIL'}] {f}　（{why}）")
            bad += 0 if ok else 1
        onnx = [n for n in names if "rtmpose-x_" in n and n.endswith(".onnx")]
        print(f"  [{'PASS' if onnx else 'FAIL'}] workspace/models/rtmpose-x_*.onnx")
        bad += 0 if onnx else 1

        # 环境的完整性：python.exe 有、torch 的 DLL 有、标准库有。
        # 只查 python.exe 是不够的——一个空的 python.exe 也能"存在"。
        for f, why in (("python/Lib/os.py", "标准库"),
                       ("python/Lib/site-packages/torch/__init__.py", "torch"),
                       ("python/Lib/site-packages/smplx/__init__.py", "smplx"),
                       ("python/msvcp140.dll", "VC 运行库")):
            ok = f in names
            print(f"  [{'PASS' if ok else 'FAIL'}] {f}　（{why}）")
            bad += 0 if ok else 1

        for n in sorted(names):
            why = None
            if any(s in n for s in FORBID_ANY):
                why = "整棵都不该在"
            elif any(s in n for s in FORBID_NAME):
                why = "不打（见文件头的说明）"
            elif any(n.startswith(s) and n != s for s in FORBID_CONTENT):
                why = "这是素材/产物，不是软件的一部分"
            if why:
                print(f"  [FAIL] 不该在包里：{n}　（{why}）")
                bad += 1

        empties = [n for n in names if n.endswith("/")]
        print(f"  [PASS] {len(empties)} 个空目录条目（后端要写的地方）")
    print(f"\n{'✓ 包是完整的' if not bad else f'✗ {bad} 处不对'}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打「数据目录」压缩包")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 zip 路径")
    ap.add_argument("--dry", action="store_true", help="只列清单和体积，不写文件")
    ap.add_argument("--fast", action="store_true", help="压缩级别 1")
    ap.add_argument("--pipeline-env", default=str(PIPELINE_ENV_DEFAULT),
                    help="要打进去的 Python 环境目录")
    ap.add_argument("--verify", metavar="ZIP", help="检查一个已经打好的包")
    args = ap.parse_args(argv)

    if args.verify:
        return verify(Path(args.verify))

    staging = Path(tempfile.mkdtemp(prefix="bogu-extra-"))
    try:
        items, warn = manifest(Path(args.pipeline_env))
        n_extra = pip_extras(Path(args.pipeline_env), staging / EXTRA_PIP_DIR)
        if n_extra:
            # **同一个过滤器也要走一遍**：这些是 manifest() 之后追加进来的，
            # 不走的话 pip 装出来的 __pycache__ 会整批混进包里——第一版就这么漏了
            # 136 个 .pyc 进去，还是自检把它抓出来的。
            for p in sorted((staging / EXTRA_PIP_DIR).rglob("*")):
                if p.is_file() and not (SKIP_NAMES & set(p.parts)):
                    items.append((p, f"{EXTRA_PIP_DIR}/{p.relative_to(staging / EXTRA_PIP_DIR).as_posix()}"))

        for w in warn:
            print(f"⚠ {w}", file=sys.stderr)
        total = sum(p.stat().st_size for p, _ in items)

        by_top: dict[str, int] = {}
        for p, rel in items:
            top = rel.split("/")[0] if not rel.startswith("workspace/") else "/".join(rel.split("/")[:2])
            by_top[top] = by_top.get(top, 0) + p.stat().st_size
        print("要打进去的（按顶层汇总）：")
        for k, v in sorted(by_top.items(), key=lambda kv: -kv[1]):
            print(f"  {human(v):>10}   {k}")
        print(f"  {'-' * 10}")
        print(f"  {human(total):>10}   合计 {len(items)} 个文件")

        print("\n不打的（显式排除）：")
        for s in SKIP_TREES:
            p = ROOT / s
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
            print(f"  {human(size):>10}   {s}")

        if warn:
            print("\n有必需文件缺失，拒绝打包。", file=sys.stderr)
            return 1
        if args.dry:
            print("\n--dry：没有写文件。")
            return 0

        # 磁盘空间：**在写之前**说，不要等到写了一半才报
        # "No space left on device"——那会留下一个大而无用的 .part，
        # 而且前面十几分钟的压缩全白做。
        out = Path(args.out)
        need = int(total * 0.70)          # 实测压掉 36%，留点余量
        free = shutil.disk_usage(_existing_ancestor(out.parent)).free
        if need > free:
            print(f"\n✗ 目标盘剩余 {human(free)}，估计要 {human(need)}（压缩后）。"
                  f"\n  换个盘：--out D:\\骨骼追踪-数据.zip", file=sys.stderr)
            return 1
        print(f"\n目标盘剩余 {human(free)}，估计要 {human(need)}，开始压缩…"
              f"（9.5 GB 大约 8 分钟）\n")

        write(items, out, args.fast)
        print("\n自检：")
        return verify(Path(args.out))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
