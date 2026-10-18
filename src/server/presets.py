"""参数预设：「参数与重算」那张卡片的一整套设置，存成一份 JSON。

    workspace/presets/<名字>.json

## 为什么落在盘上，不落浏览器
项目里所有参数的**唯一来源都是后端**：默认值和取值域分别从 /api/params 和
/api/cuts/params 取，前端不写死任何一份（见 convert.defaults / scene_cuts.TUNABLE
的说明）。预设是同一类东西。放 localStorage 的话，换浏览器、清一次缓存就没了，
也没法手工编辑、没法给命令行用——而"我这套参数调好了要存住"恰恰是它存在的理由。

## 一份预设装三组
它们的来源不同，但在用户眼里都是"这张卡片的设置"：

    params  「参数与重算」里所有滑杆/开关。键集 = convert._ALLOWED 那一份
    cuts    切点检测判据那 6 个。**它不参与 to_funscript**，只决定 cuts.json
    space   生成来源：世界系 / 相机系

## 载入时逐键校验，**不整份信任**
预设是盘上的普通文本文件：可以手改，也可能是旧版本存的（那时参数集不一样）。
所以走的是和 /convert 同一套白名单，认不出的键丢掉、类型不对的退回默认值，
最后**补全成完整的一份**——缺键就补默认值，而不是留半个字典让前端去猜。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import scene_cuts

from . import convert, paths

DIR = paths.WORKSPACE / "presets"

#: Windows 上文件名里不能出现的字符，外加控制字符。
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
#: Windows 的保留设备名。叫 CON 的文件在 NT 上根本建不出来。
_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
MAX_NAME = 64


def safe_name(raw: object) -> str | None:
    """预设名 -> 能当文件名用的名字。取不出合法名字时返回 None。

    名字是**用户输入的**，直接拼进路径就是路径穿越（`../../x`）。
    这里把分隔符和非法字符一律换成下划线，再挡一道保留设备名。
    """
    name = _UNSAFE.sub("_", str(raw or "")).strip()
    name = name.rstrip(". ")          # 结尾的点/空格在 Windows 上会被静默吃掉
    if not name or name in (".", ".."):
        return None
    if not name.strip("_"):
        # 原名里一个能用的字符都没有（`///`、`...`），替换完只剩一串下划线。
        # 那是个合法文件名，但用户看到的会是"___"——不如直接说这个名字取不出来。
        return None
    if name.split(".")[0].upper() in _RESERVED:
        name = "_" + name
    return name[:MAX_NAME]


def path_of(name: str) -> Path | None:
    n = safe_name(name)
    return None if n is None else DIR / f"{n}.json"


def _coerce(key: str, value: object) -> object | None:
    """按 _ALLOWED 里的类型转一次。转不动就返回 None（调用方保持默认值）。"""
    t = convert._ALLOWED[key]                          # noqa: SLF001 —— 同一层模块
    if t is bool:
        return bool(value)
    try:
        # bool 是 int 的子类，但 _ALLOWED 里两者不会混，所以不用特判
        return t(value)                                # type: ignore[call-arg]
    except (TypeError, ValueError):
        return None


def normalize(body: dict | None) -> dict:
    """请求/文件里的那份 -> 完整可用的一份。**永远是完整的**。"""
    body = body if isinstance(body, dict) else {}
    defaults = convert.defaults()

    params = dict(defaults)                            # 缺的键补默认值
    raw_params = body.get("params")
    if isinstance(raw_params, dict):
        for k, v in raw_params.items():
            if k not in convert._ALLOWED or v is None:  # noqa: SLF001
                continue                               # 认不出的键丢掉，不报错
            got = _coerce(k, v)
            if got is not None:
                params[k] = got

    # cuts 的校验/夹取已经有现成的（判据那边一直在用），不另写一份
    cuts = scene_cuts.normalize(body.get("cuts"))

    # space **不在 _ALLOWED 里**（它不是 to_funscript 的参数，而是"读哪份 npz"），
    # 所以它单独存一层，由 convert() 在 params 之外接住——见 convert.py 里
    # `params.get("space")` 那段。这里照同样的规则校验。
    space = body.get("space")
    if space not in paths.SPACES:
        space = "global"
    return {"params": params, "cuts": cuts, "space": space}


def read(p: Path) -> dict | None:
    """读一份预设并补全。**文件坏了/不是 JSON 就当没有**，不抛——
    盘上多一个坏文件不该让整个预设列表打不开。"""
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    out = normalize(raw)
    out["name"] = safe_name(raw.get("name") or p.stem) or p.stem
    out["saved_at"] = raw.get("saved_at")
    return out


def list_all() -> list[dict]:
    """全部预设，**新的在前**（刚存的就在下拉框第一项）。

    读不出来的文件仍然列进去（带一个 error 字段），但内容用默认值补全——
    静默跳过的话，用户只会看到"我存的预设不见了"，而不知道盘上那份坏了。
    """
    if not DIR.is_dir():
        return []
    out = []
    for p in DIR.glob("*.json"):
        item = read(p)
        if item is None:
            item = {"name": safe_name(p.stem) or p.stem, "saved_at": None,
                    **normalize(None), "error": "这份文件读不出来（不是合法 JSON）"}
        item["file"] = p.name
        item["mtime"] = p.stat().st_mtime
        out.append(item)
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def save(name: object, body: dict) -> dict:
    """写一份预设（同名覆盖）。名字取不出来就抛 ValueError。"""
    n = safe_name(name)
    if n is None:
        raise ValueError("预设名不能为空，且不能只由非法字符组成")
    DIR.mkdir(parents=True, exist_ok=True)
    data = normalize(body)
    data["version"] = 1
    data["name"] = n
    data["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = DIR / f"{n}.json"
    # 先写临时文件再替换：写到一半被中断的话，原来那份预设还在。
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return {**data, "file": p.name, "mtime": p.stat().st_mtime}


def delete(name: object) -> bool:
    p = path_of(name)
    if p is None or not p.is_file():
        return False
    p.unlink()
    return True
