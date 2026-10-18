"""把脚本发给真实设备（T-code v0.3，串口或 UDP）。

这个模块从 `F:\\游戏插件\\fsyncap\\device.py` 搬来，**但驱动方式不一样**，
下面每一条差异都写清楚了原因，因为它们不是风格问题，是两件事本来就不一样。

## 差异一：喂进来的是一条完整指令，不是"某个轴的当前位置"

fsyncap 那边上游是**实时采样流**：采集循环每采到一个样本就
`feed(axis, pos)`，由 `DeviceLink` 自己按 `rate`（默认 50Hz）的周期把"各轴最新值"
拼成一行写出去。所以它必须自己定频率、自己拼行。

这边上游是**已经算好的脚本**：`to_funscript.py` 直接把 TCode 行写成了文件，
一行就是一个采样时刻的全部轴，**而且行尾自带 `I<ms>`**（30fps 的产物是 `I33`）。
也就是说"多长时间走到下一条"这个信息已经在行里了。所以：

  · 不再有 `rate` / `interpolate` 两个设置——它们在这里没有对应物，
    留着只会让人以为改它能影响动作
  · 喂进来的单位是**整行**，不是逐轴

## 差异二：线程按"变了才写"，不是按固定周期重发

这一条是差异一的直接后果，也是这个模块里最容易写错的地方。

`I<ms>` 的语义是"用 ms 毫秒从当前位置走到这个位置"。1 秒 30 条、每条 `I33`，
正好首尾相接成一整段连续运动。**但如果按固定周期重发同一行**（比如 50Hz 发
`L0500I33`），设备每 20ms 就被重新命令一次"用 33ms 走到 0.500"——ramp 被反复
重置，实际永远走不到，表现为**设备一直落后于画面**。fsyncap 没这个问题是因为
它自己算 `I` 等于自己的发送周期，两边天然一致。

所以这里的线程是**边沿触发**：`feed()` 只把最新那一行放下并叫醒线程，线程发现
有新行就写一次，没变就什么都不做。节奏完全由脚本自己的帧率决定。

## 差异三：暂停时什么都不发

上一版那段"没有数据就不发"的约束在这里更成立：视频停住 → 播放头不动 →
没有新行 → 设备停在原地。这正是要的（停下来时设备跟着停，而不是继续跑）。

## 保留没变的部分

  · 值域：`L0500` 是 0.50，**1.0 表达不出来**（先夹到 9999 再打四位数字）
  · 写在自己的线程上：串口一次 write 可以阻塞到驱动愿意返回为止，
    绝不能发生在 Flask 的请求线程里（那会连带把整个后端卡住）
  · 断开**不给"安全位置"**：那会是一次用户没要求的全行程移动，
    而且恰好发生在他要求停下来的那一刻。停在原地是安静的那个选项
  · 端口挂了由线程自己发现（`open` 是从**线程是否活着**推出来的，不是存下来的
    一个标志）——拔掉串口时标志会一直说"连着"，而实际一个字节都没发出去
"""
from __future__ import annotations

import re
import threading

#: 六个轴，按值得给用户看的顺序。L0 是行程（"相对使用者向上"），
#: 其余几乎都是为它写的；L1/L2 只有六轴机型才有，R0/R1/R2 是扭转/翻滚/俯仰。
AXES: tuple[str, ...] = ("L0", "L1", "L2", "R0", "R1", "R2")

#: T-code 能表达的最大值（`0.` 后面那四位数字）。1.0 表达不出来，
#: 所以这是天花板，值会夹到它。
TC_MAX = 9999

DEFAULT_BAUD = 115200
#: 设备听 UDP 的地方。这是约定不是猜测：固件绑 UDP 8000，收的字节和串口上完全一样。
DEFAULT_UDP_PORT = 8000

#: 允许写进设备的整行。**白名单而不是"删掉不要的字符"**：这一行最终会变成
#: 真实硬件的一次运动，而它来自网络请求体——把任意字符串透传给设备是
#: 一个不该开的口子。格式和 `to_funscript.py` 写出来的完全一致：
#:   一或多个 `L0`/`R2` 之类的轴命令 + 值数字，行尾可选一个 `I<毫秒>`。
#:
#: 轴之间只认**空格和制表符**，不是 `\s`：`\s` 包含换行，于是
#: `"L05000\nL15000"` 会被判成合法，而它在线上就是**两条命令**——设备会先后
#: 执行两个位置，中间那一下是用户没要求的。`fullmatch` 挡不住这个，
#: 问题出在分隔符本身。
_LINE_RE = re.compile(r"[LR]\d\d{1,4}(?:[ \t]+[LR]\d\d{1,4})*(?:[ \t]+I\d{1,5})?")


def valid_line(line: str) -> bool:
    """这一行能不能写进设备。见 `_LINE_RE`。"""
    return bool(_LINE_RE.fullmatch((line or "").strip()))


def list_ports() -> list[dict]:
    """这台机器上的串口，按界面的选择框需要的样子给。

    `hwid` 一并带出去：光靠描述分不开两个一模一样的 USB 转串口线，而认错那一个
    的结果是"设备在动，动的是另一个插着的东西"。

    没装 pyserial 时返回空表而不是抛异常——设备连接是**可选**功能，
    缺一个可选依赖不该让整个后端起不来。
    """
    try:
        from serial.tools import list_ports as lp
    except Exception:                                     # noqa: BLE001
        return []
    out = []
    for p in lp.comports():
        out.append({"device": p.device, "label": p.description or p.device,
                    "hwid": getattr(p, "hwid", "") or ""})
    out.sort(key=lambda r: r["device"])
    return out


class DeviceLink:
    """一个设备，和那条一直喂它的线程。

    **刻意只有一个**。一条管线上两个设备是以后可以做的事，现在做就得先决定
    "第二个轴是什么意思"——而第一个还没被驱动过。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: 有新行要写。`feed()` set 它，线程等它——于是"有没有新东西"这件事
        #: 不需要轮询，也不会在没数据时空转。
        self._wake = threading.Event()
        self._line = ""            #: 最新喂进来的一行
        self._dirty = False        #: 上面那行还没写出去
        self._cfg: dict = {}
        self._error = ""
        self._sent = 0
        self._last_line = ""

    # -- 读 -------------------------------------------------------------------
    def status(self) -> dict:
        """界面画那个圆点和面板要的全部东西。

        `open` 是从**线程还活着**推出来的，不是存下来的一个标志：端口中途被拔掉
        会自己把线程弄死，而"打开时置位"的标志会一直说连着，实际一个字节都没发。
        """
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {"open": alive, "error": self._error, "sent": self._sent,
                    "last_line": self._last_line,
                    # 喂进来但还没写出去的那一行。界面用它区分"命令已经变了"
                    # 和"设备还没收到"——设备不动的时候先看的就是这个差。
                    "pending": self._line if self._dirty else "",
                    "config": dict(self._cfg)}

    # -- 写 -------------------------------------------------------------------
    def open(self, cfg: dict) -> dict:
        """开始发。**先关掉上一个连接**，所以这个函数同时也是"重连"。"""
        self.close()
        clean = {
            "transport": "udp" if str(cfg.get("transport")) == "udp" else "serial",
            "port": str(cfg.get("port") or ""),
            "baud": int(cfg.get("baud") or DEFAULT_BAUD),
            "host": str(cfg.get("host") or ""),
            "udp_port": int(cfg.get("udp_port") or DEFAULT_UDP_PORT),
        }
        if clean["transport"] == "serial" and not clean["port"]:
            with self._lock:
                self._error = "没有选串口"
                self._cfg = clean
            return self.status()
        if clean["transport"] == "udp" and not clean["host"]:
            with self._lock:
                self._error = "没有填设备地址"
                self._cfg = clean
            return self.status()

        with self._lock:
            self._cfg = clean
            self._error = ""
            self._sent = 0
            self._last_line = ""
            # **开局不留旧行**：留着的话一打开就会把"上一次跑到哪"当成命令发出去，
            # 通常是一次用户没要求的全回程。
            self._line = ""
            self._dirty = False
        self._stop = threading.Event()
        self._wake.clear()
        self._thread = threading.Thread(target=self._run, args=(clean, self._stop),
                                        name="bogu-device", daemon=True)
        self._thread.start()
        return self.status()

    def close(self) -> dict:
        """停止发送，**设备留在原地**。

        离开时不命令任何位置。一个"安全位置"总得挑一个，而那就是一次用户没要求
        的全行程移动，还恰好发生在他要求停下的那一刻。停在行程中间是安静的那个选项。
        """
        thread, self._thread = self._thread, None
        if thread is not None:
            self._stop.set()
            self._wake.set()          # 线程可能正阻塞在 wait 上，得叫醒它才看得到 stop
            thread.join(timeout=1.5)
        with self._lock:
            self._dirty = False
        return self.status()

    def feed(self, line: str) -> bool:
        """喂一条要发的整行。**只赋值 + 叫醒线程，绝不阻塞。**

        调用方是前端的播放循环（30Hz）。一个赋值和两次置位，别的什么都不做——
        格式化、端口、节奏全在另一半。

        返回值只说"收下了没有"：格式不对的行直接丢掉并返回 False，
        调用方（HTTP 接口）据此可以报 400。丢掉的**不动 `_line`**——
        用一行坏数据去清掉一条好命令，会让设备停在一个谁都没要求的位置。
        """
        line = (line or "").strip()
        if not valid_line(line):
            return False
        with self._lock:
            if line == self._line:
                return True           # 没变就什么都不做：见文件头"差异二"
            self._line = line
            self._dirty = True
        self._wake.set()
        return True

    # -- 线程 -----------------------------------------------------------------
    def _run(self, cfg: dict, stop: threading.Event) -> None:
        sink = None
        try:
            sink = self._connect(cfg)
            while not stop.is_set():
                # 0.2s 只是为了让 stop 有个兜底；正常路径是 feed() 把它叫醒。
                if not self._wake.wait(0.2):
                    continue
                self._wake.clear()
                if stop.is_set():
                    break
                with self._lock:
                    if not self._dirty:
                        continue
                    line, self._dirty = self._line, False
                if not line:
                    continue
                self._write(sink, line + "\n")
                with self._lock:
                    self._sent += 1
                    self._last_line = line
        except Exception as e:                            # noqa: BLE001
            with self._lock:
                self._error = f"{type(e).__name__}: {e}"
        finally:
            try:
                if sink is not None:
                    sink.close()
            except Exception:                             # noqa: BLE001
                pass

    def _connect(self, cfg: dict):
        if cfg["transport"] == "udp":
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # connect() 而不是每次 sendto()：连上之后对端地址固定，发送不用反复解析，
            # 而且网络不可达会在这里就报出来（越早越好，那是个配置错误不是运行时错误）。
            s.connect((cfg["host"], cfg["udp_port"]))
            return s
        import serial
        return serial.Serial(cfg["port"], cfg["baud"], timeout=0.2, write_timeout=0.2)

    def _write(self, sink, text: str) -> None:
        # UDP socket 有 sendto/send，串口有 write；唯一要保证的是**字节真的出去了**。
        sock = getattr(sink, "send", None)
        if callable(sock):
            sock(text.encode("ascii", "ignore"))
            return
        sink.write(text.encode("ascii", "ignore"))
