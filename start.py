"""一键起本地开发环境：Flask 后端 + Vite 前端。

    python start.py                 # 起服务，健康检查通过后打开浏览器
    python start.py --no-browser    # 不起浏览器（我自己开）
    python start.py --backend-port 5001
    python start.py --ofs           # 额外起 OFS 桥接（连 OpenFunscripter 的 Simulator3D）

Windows 上双击 start.bat 等同于此脚本。

## 为什么要有这一层

要起的东西有两个进程、端口各自固定、而且**依赖关系是有方向的**：前端只是个
Vite 外壳，所有数据（素材索引、运动数据、funscript、重算、作业队列）都来自
Flask。所以顺序和"起没起来"的判断不能靠感觉——脚本会等 `/api/health` 真的
返回才打开浏览器。

## 几条刻意的选择

1. **端口占用不是错误，是"别重复起"**。已经有一个后端在 5000 上听着时，
   再起一个只会报 EADDRINUSE 或悄悄抢不到端口，然后你对着一个**旧进程**
   调半天不出效果（这个坑本项目踩过：改了后端代码没重启，前端表现像没改）。
   所以这里先探测，占用了就跳过并打印占用者 PID 和杀掉它的命令。

2. **直接跑 vite 的入口，不经过 `npm run dev`**。项目自己的 `dev` 脚本用
   concurrently 同时起 vite 和 OFS 桥接，而桥接（web/server/index.ts，8080）
   只服务"连 OpenFunscripter 桌面端"这一个用途，起不来会把 vite 一起带走
   （concurrently -k）。前端是主线，不该被一个可选件拖下水。要桥接加 --ofs。

3. **不收尸就等于没停**。子进程和本脚本在同一个控制台组里，所以 Ctrl+C 会同时
   到达所有进程；但 `cmd /c npm run …` 这种中间层被杀掉时，它底下的 node 会被
   留在后台继续占端口（实测 8080 就是这么"停不掉"的）。所以退出时一律按
   进程树 `taskkill /F /T`——细节见 kill_all。

4. **start.bat 必须是纯 ASCII**。cmd.exe 按 OEM 代码页（中文 Windows 是 936）
   解析 .bat 的字节，不认 UTF-8：中文注释会被解码成乱码并把换行吃掉，下一行
   命令被粘到乱码后面，报成 "'ckend-port' 不是内部或外部命令"。所以那个文件里
   只留英文，解释都放这儿。
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
WEB = ROOT / "web"
LOGS = ROOT / "workspace" / "logs"

# 与 src/batch_run.py 里的 PY_GVHMR 是同一个约定：作业队列里 GVHMR 那两步
# （gvhmr / export）必须跑在那个 env 里（numpy 锁 1.23.5），其余步骤用起
# 这个服务的解释器。这里只做存在性提醒，不参与启动——真正的分派在 batch_run。
PY_GVHMR = Path("E:/anaconda3/envs/gvhmr/python.exe")

UI_PORT = 5173      # vite.config.ts 里 strictPort: true，改这里没用
OFS_PORT = 8080

_print_lock = threading.Lock()
_procs: list[tuple[str, subprocess.Popen]] = []


def _fix_console_encoding() -> None:
    """让输出在任何入口下都不炸、也不花。

    Windows 上 Python 的 stdout 编码由控制台代码页/locale 决定，两个入口不一样：
      * cmd.exe 默认 936（GBK）：中文正常，但 `✓` 这类符号编不出来，直接抛
        UnicodeEncodeError 把脚本打死（第一次跑就是这么挂的）。start.bat 里
        已经 chcp 65001，但从别处调 python start.py 就没了这层保护。
      * git-bash（MSYS）是 UTF-8 终端，可 Python 还是按 cp936 编，中文全成乱码。
    所以：判断是 MSYS 终端就钉死 UTF-8；否则保留原编码、只把编不出的字符降级，
    宁可少一个符号，不能整个脚本挂掉。
    """
    utf8_term = bool(os.environ.get("MSYSTEM")) or os.environ.get("TERM") == "xterm"
    for s in (sys.stdout, sys.stderr):
        try:
            if utf8_term:
                s.reconfigure(encoding="utf-8")
            else:
                s.reconfigure(errors="replace")
        except (AttributeError, OSError):     # 不是 TextIOWrapper（被重定向到别的东西）
            pass


def say(msg: str = "") -> None:
    with _print_lock:
        print(msg, flush=True)


def _head(title: str) -> None:
    say(f"\n=== {title} ===")


# --------------------------------------------------------------------------
# 探测
# --------------------------------------------------------------------------
def port_busy(port: int) -> bool:
    """端口有没有人在听。

    **必须 IPv4/IPv6 都试**：vite 绑的是 `localhost`，在 Windows 上解析成 ::1，
    只连 127.0.0.1 会得到"没人听"，于是这里放行、vite 随后自己报
    "Port 5173 is already in use" 退出——检测就白做了（第一次跑就是这样）。
    """
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, port), 0.4):
                return True
        except OSError:
            continue
    return False


def pid_on_port(port: int) -> str | None:
    """从 netstat 里找监听该端口的 PID。找不到返回 None（不算错误）。

    **不加 `-p TCP`**：那个过滤只留 IPv4，而 vite 绑 localhost 时是 IPv6 的
    `[::1]:5173`，加了过滤就查不到 PID，提示里只能打 `<pid>`。
    """
    try:
        out = subprocess.run(["netstat", "-ano"],
                             capture_output=True, text=True, errors="replace").stdout
    except OSError:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING" \
                and parts[1].endswith(f":{port}"):
            return parts[4]
    return None


def wait_http(url: str, what: str, timeout: float) -> bool:
    """轮询到 HTTP 200 才算起来。Flask 首次 /api/health 要扫一遍素材索引，
    几秒是正常的，所以打点让等待可见，而不是静默卡住。"""
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 400:
                    say(f"  {what} 就绪（{time.time() - t0:.1f}s）")
                    return True
        except Exception:                                    # noqa: BLE001
            pass
        if time.time() - t0 > timeout:
            say(f"  {what} 等了 {timeout:.0f}s 还没就绪：{url}")
            return False
        say("  .")
        time.sleep(1.0)


# --------------------------------------------------------------------------
# 子进程
# --------------------------------------------------------------------------
def spawn(name: str, cmd: list[str], cwd: Path, env_extra: dict[str, str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    env = {**os.environ, **env_extra}
    f = open(log, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", bufsize=1)
    _procs.append((name, p))
    threading.Thread(target=_pump, args=(name, p, f), daemon=True).start()
    say(f"  已启动 {name}（pid {p.pid}）→ {log.relative_to(ROOT)}")


def _pump(name: str, p: subprocess.Popen, f) -> None:
    assert p.stdout is not None
    for line in p.stdout:
        f.write(line)
        f.flush()
        with _print_lock:
            sys.stdout.write(f"[{name}] {line}")
            sys.stdout.flush()
    f.close()


def kill_all() -> None:
    """按**进程树**收掉所有子进程。

    两个都必须带 `/T`，而且不能只靠 Popen.terminate()：
      * `cmd /c npm run dev:sidecar` 这层里 terminate() 杀的是 cmd.exe，
        npm/node 会被留在后台继续占着 8080——实测"停不掉"就是这么来的；
      * 反过来，先 terminate 再等，往往**立刻**就"停成功"了（cmd 秒退），
        于是永远不会走到补救那一步。所以别分两步，直接树杀。
    也不先试不带 /F 的"优雅"版本：Windows 上 console 程序没有消息泵，
    不带 /F 的 taskkill 对它们基本无效，只是白等 3 秒。
    """
    for name, p in _procs:
        if p.poll() is not None:
            continue
        say(f"  停止 {name}（pid {p.pid}）")
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    for name, p in _procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            say(f"  {name} 没能停下来（pid {p.pid}），手动 taskkill /F /T /PID {p.pid}")
    _procs.clear()


# --------------------------------------------------------------------------
# 前置检查
# --------------------------------------------------------------------------
def preflight() -> None:
    _head("环境检查")
    _fix_console_encoding()
    if sys.version_info < (3, 10):
        raise SystemExit(f"需要 Python 3.10+（后端用了 `float | None` 这种写法），"
                         f"当前 {sys.version.split()[0]}")
    say(f"  python {sys.version.split()[0]}（{sys.executable}）")
    try:
        import flask  # noqa: F401
    except ImportError:
        raise SystemExit("  缺 flask：pip install -r requirements.txt "
                         "（另需 flask-cors、numpy、opencv-python）")
    say("  flask ok")

    if not PY_GVHMR.exists():
        say(f"  警告：{PY_GVHMR} 不存在——作业队列里 gvhmr/export 两步会失败，"
            f"看素材/调参不受影响")

    if shutil.which("node") is None:
        raise SystemExit("  找不到 node，先装 Node.js 18+")
    say(f"  node {subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip()}")

    vite_bin = WEB / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite_bin.exists():
        say("  web/node_modules 不齐，先装前端依赖…")
        rc = subprocess.run(["cmd", "/c", "npm", "install"], cwd=str(WEB)).returncode
        if rc != 0 or not vite_bin.exists():
            raise SystemExit("  npm install 没成功，手动在 web/ 里跑一次看看报什么")
    say("  vite ok")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="骨骼追踪 本地启动器")
    ap.add_argument("--backend-port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="不起浏览器")
    ap.add_argument("--ofs", action="store_true",
                    help="额外起 OFS 桥接（8080，连 OpenFunscripter 的 Simulator3D）")
    ap.add_argument("--timeout", type=float, default=120.0, help="等待就绪的秒数")
    args = ap.parse_args()

    print("骨骼追踪 — 本地开发环境")
    preflight()

    backend_url = f"http://{args.host}:{args.backend_port}"
    ui_url = f"http://localhost:{UI_PORT}"

    # ---- 后端起不起 ----
    _head("后端（Flask）")
    backend_started = False
    if port_busy(args.backend_port):
        pid = pid_on_port(args.backend_port)
        say(f"  {args.backend_port} 已经有进程在监听"
            + (f"（pid {pid}）" if pid else "") + "，跳过启动。")
        say(f"  ! 注意：那是个**旧进程**，改过后端代码的话它不会自动生效。")
        say(f"    要换掉它：taskkill /F /T /PID {pid or '<pid>'}")
    else:
        spawn("后端", [sys.executable, str(SRC / "serve.py"),
                       "--host", args.host, "--port", str(args.backend_port)],
              cwd=ROOT,
              # 中文日志在 GBK 控制台上会 UnicodeEncodeError；不缓冲是为了
              # werkzeug 的请求日志能实时打出来。
              env_extra={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        backend_started = True

    # ---- 前端起不起 ----
    _head("前端（Vite）")
    ui_started = False
    if port_busy(UI_PORT):
        pid = pid_on_port(UI_PORT)
        say(f"  {UI_PORT} 已经被占用"
            + (f"（pid {pid}）" if pid else "") + "，跳过启动。")
        say(f"    要换掉它：taskkill /F /T /PID {pid or '<pid>'}")
    else:
        # 直接跑 vite 入口，不经过 npm：少一层 shell，输出不缓冲，也不用
        # 为一条命令维护一棵进程树。（前端依赖装在 web/node_modules，见 preflight）
        spawn("前端", ["node", str(WEB / "node_modules" / "vite" / "bin" / "vite.js")],
              cwd=WEB,
              # vite.config.ts 读这个变量决定 /api 代理到哪。端口改了它得跟着改，
              # 否则前端还在往 5000 打，而后端在别处听着。
              env_extra={} if args.backend_port == 5000
              else {"BOGU_BACKEND": backend_url})
        ui_started = True

    # ---- OFS 桥接（可选）----
    ofs_state = "未启动（用 --ofs 开）"
    if args.ofs:
        _head("OFS 桥接")
        if port_busy(OFS_PORT):
            ofs_state = f"{OFS_PORT} 已被占用，跳过"
            say(f"  {OFS_PORT} 已被占用，跳过。")
        else:
            # 这个走 npm：入口是 TypeScript，要用 devDependency 里的 tsx 来跑。
            spawn("OFS桥接", ["cmd", "/c", "npm", "run", "dev:sidecar"], cwd=WEB,
                  env_extra={})
            ofs_state = f"http://localhost:{OFS_PORT}"

    # ---- 就绪 + 收尾 ----
    _head("等待就绪")
    ok = True
    if backend_started:
        ok &= wait_http(f"{backend_url}/api/health", "后端", args.timeout)
    else:
        # 复用了别人的实例：至少确认它真的在答话。只报"跳过启动"就宣称成功，
        # 会和"端口被某个半死的进程占着"混在一起分不清。
        alive = wait_http(f"{backend_url}/api/health", "后端（已存在的实例）", 3.0)
        say("    它答话了，但**不是这次起的**——改了后端代码要重启它才生效。"
            if alive else "    它在监听但不回 HTTP，八成是卡死或半启动，建议杀掉重来。")
        ok &= alive
    if ui_started:
        ok &= wait_http(ui_url, "前端", args.timeout)
    else:
        ok &= wait_http(ui_url, "前端（已存在的实例）", 3.0)

    if not ok:
        say("\n有服务没起来，日志见 workspace/logs/（下面也还在实时打）。")
        say("按 Ctrl+C 退出。")
    else:
        say(f"""
  前端  {ui_url}
  后端  {backend_url}/api/health
  日志  workspace/logs/（后端.log / 前端.log 实时写）
  OFS 桥接  {ofs_state}

  Ctrl+C 退出（三个进程一起停）。""")
        if not args.no_browser:
            webbrowser.open(ui_url)

    # 盯着子进程：谁先死就整体收摊，避免留半个栈在后台跑
    try:
        while True:
            time.sleep(0.5)
            for name, p in list(_procs):
                if p.poll() is not None:
                    say(f"\n{name} 退出了（code {p.returncode}），其余的一起停。")
                    return 1
    except KeyboardInterrupt:
        say("\n收到 Ctrl+C，正在停止…")
        return 0
    finally:
        kill_all()


if __name__ == "__main__":
    raise SystemExit(main())
