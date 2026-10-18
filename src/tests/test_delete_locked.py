"""删素材：原片被占用时不许留下半删状态。

    python src/tests/test_delete_locked.py

为什么单独一个文件：这条路径上的错误**只在真实占用时才出现**，而它的失败方式
特别难看——原片删在产物之后，异常一冒泡就留下"产物删干净、原片还在"的素材，
用户看到的是一条点不开也没产物的东西，比直接失败难懂得多。所以要盯住三件事：

  1. 原片被占用 -> **整件事停下**，产物一个都不许少（第 2 组）
  2. 报错要能指路（说出了"占用"、说了怎么办）——不能只丢一个 WinError 32
  3. 松手之后能删干净（第 3 组）

占用怎么模拟：Windows 上 Python 的 `open()` 拿到的句柄不共享 delete 权限，
所以只要文件开着就 `unlink` 不了（WinError 32）。这正是浏览器流式读视频时
的情形。
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from server import cleanup, index, paths  # noqa: E402
from server.app import app  # noqa: E402

SLUG = "ZTEST-DelLock"
VIDEO = paths.VIDEOS / f"{SLUG}-测试用.mp4"
PROC = paths.PROCESSED / f"{SLUG}_1280p.mp4"
OUTDIR = paths.OUTPUTS / SLUG
FSDIR = paths.FUNSCRIPTS / SLUG

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def cleanup_all():
    for p in (VIDEO, PROC):
        p.unlink(missing_ok=True)
    for d in (OUTDIR, FSDIR):
        shutil.rmtree(d, ignore_errors=True)


print("\n[0] 造一条假素材")
cleanup_all()
VIDEO.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=c=blue:s=64x64:d=1:r=10", str(VIDEO)], check=False)
PROC.write_bytes(b"x" * 1024)
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "hmr4d_results_incam.npz").write_bytes(b"x" * 2048)
FSDIR.mkdir(parents=True, exist_ok=True)
(FSDIR / "L0.funscript").write_text('{"actions":[]}', encoding="utf-8")
index.scan()
check("索引里能找到它", index.by_slug(SLUG) is not None)
check("视频文件是真的", VIDEO.exists() and VIDEO.stat().st_size > 0,
      f"{VIDEO.stat().st_size if VIDEO.exists() else 0} 字节")

client = app.test_client()

print("\n[1] 正常删除（没有占用）应当成功")
PROC.unlink()          # 先不测这一半，第 3 组再测
r = client.delete(f"/api/clips/{SLUG}?with_video=1")
check("返回 200", r.status_code == 200, str(r.status_code))
check("原片没了", not VIDEO.exists())
check("产物没了", not OUTDIR.exists() and not FSDIR.exists())
body = r.get_json() or {}
check("回执里有 removed", bool(body.get("removed")), str(body.get("removed")))

print("\n[2] 原片被占用：整件事停下，产物一个都不许少")
cleanup_all()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=c=red:s=64x64:d=1:r=10", str(VIDEO)], check=False)
PROC.write_bytes(b"x" * 1024)
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "hmr4d_results_incam.npz").write_bytes(b"x" * 2048)
FSDIR.mkdir(parents=True, exist_ok=True)
(FSDIR / "L0.funscript").write_text('{"actions":[]}', encoding="utf-8")
index.scan()

with open(VIDEO, "rb"):                       # 句柄占住，模拟浏览器在读
    r = client.delete(f"/api/clips/{SLUG}?with_video=1")
    check("返回 409（不是 500）", r.status_code == 409, str(r.status_code))
    msg = (r.get_json() or {}).get("error", "")
    check("错误里点明了是占用", "占用" in msg, msg[:60] + "...")
    check("错误里给了下一步怎么办", "暂停" in msg and "取消勾选" in msg)
    check("错误不是一坨 HTML", "<!doctype" not in msg.lower())
    # 这三条是重点：失败必须等于什么都没做
    check("产物目录还在（没被先删掉）", OUTDIR.exists())
    check("脚本目录还在", FSDIR.exists())
    check("预处理视频还在", PROC.exists())
    check("原片也还在", VIDEO.exists())

print("\n[3] 松手之后再删：应当删干净")
r = client.delete(f"/api/clips/{SLUG}?with_video=1")
check("返回 200", r.status_code == 200, str(r.status_code) + str(r.get_json()))
check("原片没了", not VIDEO.exists())
check("产物目录没了", not OUTDIR.exists())
check("脚本目录没了", not FSDIR.exists())
check("预处理视频没了", not PROC.exists())
check("索引里也查不到了", index.by_slug(SLUG) is None)

print("\n[4] 删失败的异常也要是 JSON（不然界面只会显示一句没用的 HTML）")
with open(VIDEO, "rb") if VIDEO.exists() else open(Path(__file__), "rb"):
    pass
r = client.get("/api/clips/__不存在的素材__/usage")
check("404 是 JSON", r.status_code == 404 and "error" in (r.get_json() or {}),
      str(r.get_json()))

cleanup_all()
print(f"\n===== {len(passes)} 通过, {len(fails)} 失败 =====")
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
