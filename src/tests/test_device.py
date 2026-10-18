"""设备连接（T-code 发送）的验证。

    python src/tests/test_device.py

**不接硬件**：用一个绑在 127.0.0.1 上的 UDP 套接字当假设备，`DeviceLink` 往它
发，这里收。这样能验的恰好是那些"错了也不报错"的地方：

  1. **行的白名单**。这一行最终会变成真实硬件的一次运动，而它来自请求体。
     放任意字符串过去，一个手滑就是设备乱动。
  2. **去重**。同一行连发两次只能出去一次——`I<ms>` 是"用 ms 毫秒走到这里"，
     重发会把那段 ramp 从头开始，设备于是永远走不到（详见 device.py 文件头）。
     这一条错了的表现是"设备一直在动，但总是慢半拍"，最难查。
  3. **合并**。同一瞬间喂好几行，出去的**只有最后那一行**，不能排队全发。
  4. **开局不发**。连上就发"上一次停在哪"通常是一次用户没要求的全回程。
  5. **断开不动设备**。断开时一个字节都不许发。
  6. **状态是从线程推的**。线程被弄死之后 `open` 必须变成 False（拔串口就是
     这个情形），而存下来的标志会一直说"连着"。
"""
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from server.device import DeviceLink, valid_line   # noqa: E402

FAILS: list[str] = []


def check(cond, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


class FakeDevice:
    """一个 UDP 假设备：收字节，按到达顺序存下来。"""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.45)
        self.port = self.sock.getsockname()[1]
        self.lines: list[str] = []
        #: 累计收到多少条。`lines` 会被 `settle()` 清空，要问"总共发了几条"时用它。
        self.total = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            got = [x for x in data.decode("ascii").split("\n") if x]
            self.lines.extend(got)
            self.total += len(got)

    def settle(self, seconds: float = 0.35) -> list[str]:
        """等一会儿，把**这一次**收到的行读出来。

        **收完就清空**：不清的话第二次调用会把上一步的行一起返回，
        而下面每一处断言问的都是"这一段里发了什么"。这个 bug 我在测试里先踩了
        一次——5 项失败里有 4 项是它（不是被测代码的问题）。
        """
        time.sleep(seconds)
        got = [x for x in self.lines if x]
        self.lines.clear()
        return got

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


def cfg_for(dev: FakeDevice) -> dict:
    return {"transport": "udp", "host": "127.0.0.1", "udp_port": dev.port}


def test_valid_line() -> None:
    print("\n[1] 行白名单：只有脚本自己写得出那种行才放行")
    for good in ("L05000", "L05000 L15000 R05000 I33",
                 "L00000 L10000 L20000 R00000 R10000 R20000 I33"):
        check(valid_line(good), f"放行 {good!r}")
    for bad in ("", "   ", "hello", "L05000; rm -rf /", "M05000",
                "L05000\nL15000", "L05000 I", "I33", "L05000 I33 I33",
                "L0500012345", "l05000"):
        check(not valid_line(bad), f"拦下 {bad!r}")


def test_feed_and_dedupe() -> None:
    dev = FakeDevice()
    link = DeviceLink()
    try:
        print("\n[2] 连上之后什么都不发")
        st = link.open(cfg_for(dev))
        check(st["open"], "open=True")
        check(dev.settle() == [], "一个字节都没发（不许把上次的位置当命令发出去）")

        print("\n[3] 喂一行 -> 原样出去")
        link.feed("L05000 L15000 I33")
        got = dev.settle()
        check(got == ["L05000 L15000 I33"], f"收到 {got}")

        print("\n[4] 同一行再喂几次 -> 只出去一次")
        before = dev.total
        for _ in range(5):
            link.feed("L05000 L15000 I33")
        check(dev.settle() == [], "重复的行被去掉了（重发会把 I 的 ramp 从头开始）")
        check(dev.total == before, "累计条数没有变多")

        print("\n[5] 同一瞬间喂好几行 -> 只出去最后那一行")
        for v in range(1000, 1050):
            link.feed(f"L0{v:04d} I33")
        got = dev.settle()
        check(len(got) == 1, f"中间那些被合并掉了（收到 {len(got)} 条：{got}）")
        check(got and got[0].startswith("L01049"), f"留下的是最后那一行：{got}")
        st = link.status()
        check(st["sent"] == 2, f"已发计数只算真写出去的（{st['sent']}）")
        check(dev.total == st["sent"], f"和设备实际收到的条数对得上（{dev.total}）")
        check(st["last_line"] == "L01049 I33", f"命令(last_line) = {st['last_line']!r}")
        check(st["pending"] == "" or st["pending"] == st["last_line"],
              f"没有待发的东西了（pending={st['pending']!r}）")

        print("\n[6] 非法行：丢掉，而且**不动**已经生效的那一行")
        check(link.feed("garbage; reboot") is False, "非法行被拒绝（接口据此报 400）")
        check(link.feed("") is False, "空行被拒绝")
        check(dev.settle() == [], "非法行没有被写出去")
        check(link.status()["last_line"] == "L01049 I33",
              "丢弃非法行不许清掉当前命令（否则设备会停在一个谁都没要求的位置）")

        print("\n[7] 断开：设备留在原地，一个字节都不发")
        before = dev.total
        link.close()
        check(dev.settle() == [], "断开时不发任何'安全位置'")
        check(link.status()["open"] is False, "断开后 open=False")
        check(dev.total == before, "没有多出来的字节")

        print("\n[8] 重连之后，和上一次相同的那一行**必须发得出去**")
        link.open(cfg_for(dev))
        link.feed("L01049 I33")
        check(dev.settle() == ["L01049 I33"],
              "重连后同样的内容照样发（去重记忆不能在重连后还留着）")
        link.close()
    finally:
        link.close()
        dev.close()


def test_dead_link() -> None:
    """线程死了 -> open 必须变成 False。拔串口就是这件事。"""
    print("\n[9] open 从线程推出来，不是存下来的一个标志")
    link = DeviceLink()
    # 指向一个没人听的端口：UDP 的 connect() 不报错，但线程能起来；
    # 这里要的是"线程结束后 open 变 False"这条路径，所以直接掐线程。
    st = link.open({"transport": "udp", "host": "127.0.0.1", "udp_port": 1})
    check(st["open"], "先确认它本来是连着的")
    t = link._thread
    link._stop.set()                 # 让 _run 自己退出（相当于端口消失）
    link._wake.set()
    if t is not None:
        t.join(timeout=2.0)
    check(not t.is_alive(), "线程确实结束了")
    check(link.status()["open"] is False,
          "线程没了之后 open=False（存标志的话这里会一直说'连着'）")
    link.close()


def test_bad_config() -> None:
    print("\n[10] 配置不全 -> 直接报错，不偷偷起一个线程")
    link = DeviceLink()
    check(link.open({"transport": "serial", "port": ""})["error"] == "没有选串口",
          "没选串口")
    check(link.open({"transport": "udp", "host": ""})["error"] == "没有填设备地址",
          "没填地址")
    check(link.status()["open"] is False, "这两种情况都没有把线程起起来")
    link.close()


def test_ports_never_raises() -> None:
    print("\n[11] 串口枚举：读不出来也要给一张空表，不许抛")
    from server.device import list_ports
    ports = list_ports()
    check(isinstance(ports, list), f"返回列表（这台机器上 {len(ports)} 个）")
    if ports:
        check(all({"device", "label", "hwid"} <= set(p) for p in ports),
              "每条都带 device/label/hwid（hwid 是分开两个同型号线唯一的依据）")


def main() -> int:
    test_valid_line()
    test_feed_and_dedupe()
    test_dead_link()
    test_bad_config()
    test_ports_never_raises()
    print()
    if FAILS:
        print(f"✗ {len(FAILS)} 项没过：")
        for f in FAILS:
            print("   -", f)
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
