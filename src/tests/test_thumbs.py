"""缩略图的验证。

    python src/tests/test_thumbs.py

盯住四件事：
  1. 取样点要夹取——片长 2 秒的素材不能去抽第 60 秒
  2. **抽出来的是真能解码的 JPEG**（踩过一次：临时文件写成 `thumb.jpg.tmp`，
     ffmpeg 靠扩展名猜封装格式，`.tmp` 它不认，直接失败——手工跑同样的命令
     写 .jpg 却是好的，所以只在模块里复现）
  3. 缓存要生效：第二次不该再跑 ffmpeg（列表一次请求几十张，每次都抽就废了）
  4. 源比缩略图新时要重抽（素材被换过了）
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from server import paths, thumbs  # noqa: E402

SLUG = "ZTEST-Thumb"
PROC = paths.PROCESSED / f"{SLUG}_1280p.mp4"
OUTDIR = paths.OUTPUTS / SLUG

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def cleanup_all():
    PROC.unlink(missing_ok=True)
    shutil.rmtree(OUTDIR, ignore_errors=True)


print("\n[1] 取样点夹取")
check("正常片长取 25%", thumbs._pick_time(100.0) == 25.0, str(thumbs._pick_time(100.0)))
check("很短的素材不会去抽后面（夹到 1s）",
      thumbs._pick_time(2.0) == thumbs.MIN_AT, str(thumbs._pick_time(2.0)))
check("很长的素材不会抽太后面（夹到上限）",
      thumbs._pick_time(3600.0) == thumbs.MAX_AT, str(thumbs._pick_time(3600.0)))
check("时长未知时给一个安全值", thumbs._pick_time(None) == thumbs.MIN_AT)
check("时长 0 也不会算出 0", thumbs._pick_time(0.0) == thumbs.MIN_AT)

print("\n[2] 真抽一张：必须是能解码的 JPEG")
cleanup_all()
paths.PROCESSED.mkdir(parents=True, exist_ok=True)
# 造一段 4 秒的测试视频当"预处理产物"。用 testsrc 而不是纯色——纯色图看不出
# 抽帧位置对不对，也测不出 thumbnail 滤镜有没有吃掉帧
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                "-i", "testsrc=size=320x240:rate=30:duration=4", str(PROC)], check=False)
check("测试素材造出来了", PROC.exists() and PROC.stat().st_size > 0)

t0 = time.time()
got = thumbs.ensure(SLUG, 4.0)
dt = time.time() - t0
check("ensure 返回了路径", got is not None and got.exists(), str(got))
check("落在 outputs/<slug>/thumb.jpg 下（删除素材时跟着一起走）",
      got is not None and got.parent == OUTDIR and got.name == "thumb.jpg")
if got and got.exists():
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_name,width,height", "-of", "csv=p=0", str(got)],
                       capture_output=True, text=True)
    check("抽出来的是真能解码的图（不是半张残件）", r.returncode == 0 and "mjpeg" in r.stdout,
          r.stdout.strip() or r.stderr.strip()[:60])
    check("尺寸是正方形（竖屏素材也统一）", "128,128" in r.stdout, r.stdout.strip())
    # 临时文件不该留下来
    check("没留下 .tmp 残留",
          not list(OUTDIR.glob("*.tmp*")), str(list(OUTDIR.glob("*.tmp*"))))
    mtime1 = got.stat().st_mtime

    print("\n[3] 缓存：第二次不该再跑 ffmpeg")
    time.sleep(1.1)                       # 让 mtime 有可分辨的变化
    got2 = thumbs.ensure(SLUG, 4.0)
    check("第二次还是同一个文件", got2 == got)
    check("第二次没重抽（mtime 没变）", got2.stat().st_mtime == mtime1)

    print("\n[4] 源比缩略图新 -> 要重抽")
    time.sleep(1.1)
    PROC.touch()                          # 模拟素材被换过
    got3 = thumbs.ensure(SLUG, 4.0)
    check("重抽了（mtime 变了）", got3 is not None and got3.stat().st_mtime != mtime1)

print("\n[5] 拿不到源视频就返回 None（不抛异常）")
PROC.unlink()
check("源没了 -> None", thumbs.ensure(SLUG, 4.0) is None)

print("\n[6] 轨迹段小图：帧号要用**正在裁的那条视频**的帧率换算")
# 真发生过（2026-10-16）：人物面板里的小图裁的是别人。原因是它拿**原片**的帧率
# 去除帧号，而框的帧号是在**预处理产物**上数的——预处理会把 >40fps 的源降到
# 30fps，于是 60fps 的原片算出来的时刻正好差一倍（实测 tid 1 该在 33.07s，
# 裁成了 16.53s，正好是另一个镜头里的另一个人）。
#
# 造一段前 1.5 秒红、后 1.5 秒蓝的 30fps 预处理产物，再配一条 60fps 的原片：
# 取第 60 帧（=2.0s，蓝）。按原片帧率算会取到 1.0s（红）——颜色直接把答案说出来。
FPS_SLUG = "ZTEST-ThumbFps"
FPS_PROC = paths.PROCESSED / f"{FPS_SLUG}_1280p_30fps.mp4"
FPS_SRC = paths.VIDEOS / f"{FPS_SLUG}.mp4"
FPS_OUT = paths.OUTPUTS / FPS_SLUG
shutil.rmtree(FPS_OUT, ignore_errors=True)
paths.PROCESSED.mkdir(parents=True, exist_ok=True)
paths.VIDEOS.mkdir(parents=True, exist_ok=True)
subprocess.run(["ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=160x90:d=1.5:r=30",
                "-f", "lavfi", "-i", "color=c=blue:s=160x90:d=1.5:r=30",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
                "-pix_fmt", "yuv420p", str(FPS_PROC)], check=False)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                "-i", "color=c=green:s=160x90:d=3:r=60", "-pix_fmt", "yuv420p",
                str(FPS_SRC)], check=False)


def avg_rgb(p: Path) -> tuple[float, float, float]:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = max(1, len(out) // 3)
    return tuple(sum(out[i::3]) / n for i in range(3))


if FPS_PROC.exists() and FPS_SRC.exists():
    from server import index
    index.scan(use_cache=False)                    # 让原片（60fps）进索引
    clip = index.by_slug(FPS_SLUG)
    check("原片按 60fps 进索引（旧代码就是拿它去除的）",
          clip is not None and abs((clip.fps or 0) - 60.0) < 0.1,
          f"clip.fps={clip.fps if clip else None}")
    got = thumbs.segment_thumb(FPS_SLUG, 1, 60, [40.0, 20.0, 90.0, 70.0], size=32)
    check("小图抽出来了", got is not None and got.exists(), str(got))
    if got and got.exists():
        r, g, b = avg_rgb(got)
        check(f"第 60 帧该取 2.0s 那一段（蓝）——实际 RGB {r:.0f},{g:.0f},{b:.0f}",
              b > 150 and r < 100, f"r={r:.0f} b={b:.0f}")
        check("不是 1.0s 那一段（红）——拿原片帧率除就会落到这里", r < 150)

    print("\n[7] 框比画面还大时也要出图（crop 的边长不许超输入）")
    # 人物特写贴着画面边缘时，框可能比画面还大；而 ffmpeg 的 crop **边长**超过
    # 输入就直接报 "Invalid too big or non positive size"——小图静默地没有，
    # 界面上一片灰。x/y 为负是允许的（补边），边长不是，所以用 min(iw,…) 夹。
    big = thumbs.segment_thumb(FPS_SLUG, 2, 60, [0.0, 0.0, 160.0, 90.0], size=32)
    check("整画面那么大的框也抽得出图", big is not None and big.exists(), str(big))
else:
    check("测试素材造不出来（ffmpeg 有问题？）", False)

shutil.rmtree(FPS_OUT, ignore_errors=True)
FPS_PROC.unlink(missing_ok=True)
FPS_SRC.unlink(missing_ok=True)

cleanup_all()
print(f"\n===== {len(passes)} 通过, {len(fails)} 失败 =====")
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
