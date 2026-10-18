"""参数预设的验证。

    python src/tests/test_presets.py

这一段的每一处都是"写错了不报错"：

  1. **路径穿越**：预设名是用户输入的，直接拼进路径时 `../..` 就写到工作区外面去了。
     不能靠"界面上不会有人这么填"——名字也来自盘上的文件名（list_all 会回读）。
  2. **不整份信任盘上的文件**：预设可以手改，也可能是旧版本存的（那时参数集不一样）。
     认不出的键要丢掉、类型不对的退回默认，而且结果必须**是完整的一份**——
     留半个字典让前端去猜，表现是"加载了预设但有一半滑杆不动"。
  3. **坏文件不许带崩整个列表**：盘上多一个手改坏的 JSON，用户只会看到"我的预设
     不见了"，而不知道是那份文件坏了。

不碰真的 workspace/presets：把模块级的 DIR 指到临时目录。
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from server import convert, presets                 # noqa: E402
import scene_cuts                                   # noqa: E402

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


presets.DIR = Path(tempfile.mkdtemp()) / "presets"
DEFAULTS = convert.defaults()
BASE = {
    "params": {"highpass": 1.5, "travel_l0": 0.2, "post_order": "gain,height_limit",
               "compensate": True, "gain_r2": -1.2},
    "cuts": {"abs_min": 0.4, "min_gap": 0.3},
    "space": "incam",
}

# --------------------------------------------------------------------------
print("\n[1] 预设名 -> 文件名：路径穿越和非法字符")
check("普通名字原样保留", presets.safe_name("柔和") == "柔和", presets.safe_name("柔和"))
check("斜杠被换掉（不能跳出目录）",
      "/" not in (presets.safe_name("../../x") or ""), repr(presets.safe_name("../../x")))
check("反斜杠同样被换掉",
      presets.safe_name("..\\..\\x") == ".._.._x", repr(presets.safe_name("..\\..\\x")))
check("空名字取不出来", presets.safe_name("") is None and presets.safe_name("   ") is None)
check("只有非法字符 -> None", presets.safe_name("///") is None, repr(presets.safe_name("///")))
check("结尾的点和空格被去掉（Windows 会静默吃掉）",
      presets.safe_name("名字. ") == "名字", repr(presets.safe_name("名字. ")))
check("保留设备名被挡（CON 在 NT 上建不出来）",
      presets.safe_name("CON") == "_CON", repr(presets.safe_name("CON")))
check("  但 con.txt 这种只是恰好同名前缀 -> 不挡（要的是完整设备名）",
      presets.safe_name("CONF") == "CONF", repr(presets.safe_name("CONF")))
check("超长名字被截断", len(presets.safe_name("x" * 500) or "") == presets.MAX_NAME)
check("path_of 对非法名字给 None", presets.path_of("") is None)

# --------------------------------------------------------------------------
print("\n[2] 校验：认不出的键丢掉、缺的补默认、结果永远是完整的")
n = presets.normalize(BASE)
check("合法值原样进来", n["params"]["highpass"] == 1.5, str(n["params"]["highpass"]))
check("布尔键保持布尔", n["params"]["compensate"] is True)
check("负增益（掩蔽那种用法）不被当成非法", n["params"]["gain_r2"] == -1.2)
check("结果键集 == _ALLOWED（一个不多一个不少）",
      set(n["params"]) == set(convert._ALLOWED),
      f"{len(n['params'])} 个")
check("缺的键补的是后端默认值",
      n["params"]["travel_l1"] == DEFAULTS["travel_l1"]
      and n["params"]["max_deg"] == DEFAULTS["max_deg"],
      f"travel_l1={n['params']['travel_l1']}")
check("space 被单独存下来", n["space"] == "incam")

bad = presets.normalize({"params": {"highpass": "abc", "不认识": 1, "travel_l0": None},
                         "cuts": {"abs_min": 9999, "min_gap": -5},
                         "space": "火星系"})
check("字符串转不成数 -> 退回默认值",
      bad["params"]["highpass"] == DEFAULTS["highpass"], str(bad["params"]["highpass"]))
check("认不出的键被丢掉", "不认识" not in bad["params"])
check("值为 null 的键被跳过（退回默认）",
      bad["params"]["travel_l0"] == DEFAULTS["travel_l0"])
check("数字字符串能转（JSON 里可能是字符串）",
      presets.normalize({"params": {"highpass": "2.5"}})["params"]["highpass"] == 2.5)
check("越界的判据被夹回来（走 scene_cuts.normalize）",
      bad["cuts"]["abs_min"] == scene_cuts.LIMITS["abs_min"][1]
      and bad["cuts"]["min_gap"] == 0, str(bad["cuts"]["abs_min"]) + "/" + str(bad["cuts"]["min_gap"]))
check("非法 space 退回 global", bad["space"] == "global", str(bad["space"]))
check("body 是 None / 不是字典也不炸",
      presets.normalize(None)["params"] == DEFAULTS and presets.normalize("x")["params"] == DEFAULTS)

# --------------------------------------------------------------------------
print("\n[3] 存 / 读 / 列 一圈回来")
saved = presets.save("测试预设", BASE)
check("存下来名字对", saved["name"] == "测试预设", str(saved["name"]))
check("落盘的是一个 json 文件", (presets.DIR / "测试预设.json").is_file())
check("存的时候补了时间戳", bool(saved.get("saved_at")), str(saved.get("saved_at")))

again = presets.read(presets.DIR / "测试预设.json")
check("读回来内容一致（含三组）",
      again is not None and again["params"]["highpass"] == 1.5
      and again["cuts"]["abs_min"] == 0.4 and again["space"] == "incam")
check("读回来是完整的一份", set(again["params"]) == set(convert._ALLOWED))

lst = presets.list_all()
check("列表里能看到它", [x["name"] for x in lst] == ["测试预设"], str([x["name"] for x in lst]))
check("列表项也已经是补全过的", set(lst[0]["params"]) == set(convert._ALLOWED))
check("没有目录时返回空清单（不抛）",
      (setattr(presets, "DIR", presets.DIR / "nope") or presets.list_all() == []))
presets.DIR = presets.DIR.parent / "presets"

# 同名覆盖
presets.save("测试预设", {"params": {"highpass": 3.0}, "cuts": {}, "space": "global"})
check("同名覆盖（不是存成第二份）", len(list(presets.DIR.glob("*.json"))) == 1,
      str([p.name for p in presets.DIR.glob("*.json")]))
check("  覆盖后是新的值", presets.read(presets.DIR / "测试预设.json")["params"]["highpass"] == 3.0)
check("  没传的键退回默认，不留上一份的残留",
      presets.read(presets.DIR / "测试预设.json")["params"]["travel_l0"] == DEFAULTS["travel_l0"])

presets.save("另一份", {"params": {"highpass": 5.0}, "cuts": {}, "space": "global"})
check("再存一份 -> 列表两条", len(presets.list_all()) == 2, str(len(presets.list_all())))
try:
    presets.save("///", BASE)
    check("纯非法字符的名字应当被拒绝", False)
except ValueError:
    check("纯非法字符的名字被拒绝", True)

# --------------------------------------------------------------------------
print("\n[4] 盘上的文件坏了：不许带崩列表，也不许静默消失")
(presets.DIR / "坏掉的.json").write_text("{这不是 JSON", encoding="utf-8")
lst = presets.list_all()
names = [x["name"] for x in lst]
check("坏文件仍然出现在列表里（不静默消失）", "坏掉的" in names, str(names))
check("  并且标了 error", any(x.get("error") for x in lst if x["name"] == "坏掉的"))
check("  内容用默认值兜底（不会让前端崩）",
      set(next(x for x in lst if x["name"] == "坏掉的")["params"]) == set(convert._ALLOWED))
check("好文件不受影响",
      next(x for x in lst if x["name"] == "另一份")["params"]["highpass"] == 5.0)
(presets.DIR / "非对象.json").write_text("[1,2,3]", encoding="utf-8")
check("内容是数组也算读不出来，不当成预设用", presets.read(presets.DIR / "非对象.json") is None)

# --------------------------------------------------------------------------
print("\n[5] 删除")
check("删得掉", presets.delete("另一份") is True and not (presets.DIR / "另一份.json").exists())
check("删不存在的返回 False（不抛）", presets.delete("另一份") is False)
check("非法名字删不掉、也不会伤到别的文件",
      presets.delete("") is False and presets.delete("../../x") is False)
check("  别的预设还在", (presets.DIR / "测试预设.json").is_file())

# --------------------------------------------------------------------------
print("\n[6] 旧版本存的预设（那时键集不一样）也能用")
old = presets.DIR / "旧的.json"
old.write_text(json.dumps({"name": "旧的", "params": {"highpass": 2.0, "已经删掉的参数": 1}},
                          ensure_ascii=False), encoding="utf-8")
item = presets.read(old)
check("读得出来", item is not None)
check("  旧键保留", item["params"]["highpass"] == 2.0)
check("  删掉的键不残留", "已经删掉的参数" not in item["params"])
check("  新键补默认值", item["params"]["compensate"] == DEFAULTS["compensate"])
check("  名字从文件里的 name 字段取", item["name"] == "旧的", str(item["name"]))

print()
if fails:
    print(f"✗ {len(fails)} 项没过：")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print(f"✓ 全部通过（{len(passes)} 项）")
