"""生成 osr-emu 对比页（视频 + SR6 模拟器同步），把全部素材做进一个页面。

为什么必须走 HTTP，不能双击打开
--------------------------------
1. **`file://` 下浏览器禁止 ES 模块导入**：`<script type="module">` 的 `import` 按 CORS
   处理，`file://` 的来源是 `null`，Chromium 系直接拒绝——页面表现是**一片空白**，
   且报错只在控制台，肉眼看不到。
2. 本地 mp4 在 `file://` 下同样受限，而对比必须先能播视频。

失败要看得见
------------
页面顶部有致命错误横幅：捕获 `window.onerror`、模块加载失败，并在构造模拟器前
**先探测 WebGL**。空白页最难查，所以把原因直接显示出来。

一个页面 = 全部素材
-------------------
素材下拉框切换时会**同时换视频和 TCode 脚本**——只换其一就会出现「视频是 A、
脚本是 B」的错配，对比毫无意义。

用法
----
    # 自动扫描 workspace/funscripts 下全部素材
    python src/make_emu_page.py --scan --out ./emu_replay

    # 单条素材
    python src/make_emu_page.py <tcode.txt> --out <目录> --video <mp4>

然后（脚本会打印同样的命令）：
    python -m http.server 8765 --directory <目录>
    # 浏览器打开 http://localhost:8765/<name>.emu.html
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB_DEFAULT = ROOT / "workspace" / "emu" / "osr-emu.js"
TC_SUFFIX = ".tcode.txt"

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>骨盆动作回放 — SR6</title>
<style>
  html,body{margin:0;height:100%;background:#14161a;color:#dde3ea;overflow:hidden;
            font:13px/1.55 ui-monospace,Consolas,monospace}
  #stage{position:absolute;inset:0;display:flex}
  #left{flex:0 0 auto;height:100%;display:flex;align-items:center;justify-content:center;background:#0b0d10}
  #vid{height:100%;display:block}
  #canvas{flex:1 1 auto;height:100%;position:relative;min-width:0}
  #canvas canvas{display:block;width:100%!important;height:100%!important}
  .panel{position:fixed;background:rgba(12,14,18,.86);border:1px solid #2b3038;
         border-radius:8px;padding:10px 13px;backdrop-filter:blur(6px);z-index:5}
  #hud{left:12px;top:12px;min-width:340px}
  #ctl{left:12px;bottom:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;max-width:62vw}
  #fatal{position:fixed;left:0;right:0;top:0;background:#5a1d1d;border-bottom:2px solid #a33;
         padding:12px 16px;z-index:99;display:none;white-space:pre-wrap;color:#ffdede}
  h3{margin:0 0 7px;font-size:12px;font-weight:600;color:#8fa3b8;letter-spacing:.06em}
  table{border-collapse:collapse}
  td{padding:1px 0} td.k{color:#8fa3b8;padding-right:8px}
  td.v{color:#e8f1ff;text-align:right;min-width:58px}
  .bar{width:96px;height:7px;background:#242a33;border-radius:4px;overflow:hidden;
       display:inline-block;vertical-align:middle;margin-left:7px}
  .bar>i{display:block;height:100%;background:#4da3ff}
  button{font:inherit;background:#222833;color:#dde3ea;border:1px solid #39414f;
         border-radius:5px;padding:4px 11px;cursor:pointer}
  button:hover{background:#2c3440}
  input[type=range]{width:96px;vertical-align:middle}
  select{font:inherit;background:#222833;color:#dde3ea;border:1px solid #39414f;
         border-radius:5px;padding:3px;max-width:210px}
  .muted{color:#7d8b9c} .warn{color:#ffc46b} .big{color:#e8f1ff;font-weight:600}
</style>
</head>
<body>
<div id="fatal"></div>
<div id="stage">
  <div id="left"><video id="vid" controls playsinline preload="metadata"></video></div>
  <div id="canvas"></div>
</div>
<div class="panel" id="hud">
  <h3>骨盆动作回放</h3>
  <div id="meta"></div>
  <table id="axes"></table>
  <div id="warn" class="warn"></div>
</div>
<div class="panel" id="ctl">
  <button id="pp">播放</button>
  <button id="rs">重放</button>
  <label class="muted"><input type="range" id="sp" min="0.1" max="3" step="0.1" value="1"> <span id="spv">1.0x</span></label>
  <label class="muted">素材 <select id="csel"></select></label>
  <label class="muted">画面 <select id="vsel"></select></label>
</div>

<script>
window.__fatal = function (msg) {
  var el = document.getElementById('fatal');
  el.style.display = 'block';
  el.textContent = String(msg);
};
window.onerror = function (m, s, l, c, e) {
  window.__fatal('脚本错误: ' + m + '\n  ' + (s || '') + ':' + (l || '') +
                 (e && e.stack ? '\n' + e.stack : ''));
  return false;
};
window.addEventListener('unhandledrejection', function (ev) {
  window.__fatal('未处理的 Promise 拒绝: ' + (ev.reason && ev.reason.message || ev.reason));
});
setTimeout(function () {
  if (!window.__started) {
    window.__fatal('模块没有启动。常见原因：\n' +
      '  1) 用 file:// 直接打开——浏览器会以 CORS 拒绝加载 ES 模块（必须走 http://）\n' +
      '  2) osr-emu.js 不在同目录\n' +
      '  3) 按 F12 看控制台里 WebGL / 模块相关报错');
  }
}, 4000);
(function () {
  window.__webglOk = (function () {
    try { var c = document.createElement('canvas');
          return !!(c.getContext('webgl2') || c.getContext('webgl')); }
    catch (e) { return false; }
  })();
  if (!window.__webglOk) {
    window.__fatal('WebGL 不可用，3D 模拟器无法渲染。\n' +
      '  在 edge://gpu 检查 WebGL 状态，并开启“使用硬件加速”。');
  }
})();
</script>

<script type="module">
import { OSREmulator } from './osr-emu.js';
window.__started = true;
const $ = id => document.getElementById(id);
const CLIPS = __CLIPS__;
const AX = ['L0','L1','L2','R0','R1','R2'];

let clip = 0, lines = [], fps = 30, idx = 0;

const osr = window.__webglOk ? new OSREmulator('#canvas', { model: 'SR6' }) : null;
const write = s => { if (osr) osr.write(s + '\n'); };

$('axes').innerHTML = AX.map(k => `<tr><td class="k">${k}</td><td class="v" id="v_${k}">–</td>` +
  `<td><div class="bar"><i id="b_${k}" style="width:0%"></i></div></td></tr>`).join('');
$('csel').innerHTML = CLIPS.map((c, i) => `<option value="${i}">${c.name}</option>`).join('');

// 从 "L05000" 解析轴名与 0..1 的值（位数即精度）
function parse(line) {
  const out = {};
  for (const tok of line.trim().split(/\s+/)) {
    const m = tok.match(/^([LR])(\d)(\d{1,4})$/);
    if (m) out[m[1] + m[2]] = parseInt(m[3], 10) / (10 ** m[3].length);
  }
  return out;
}
function paint() {
  if (!lines.length) return;
  const v = parse(lines[Math.max(0, Math.min(idx, lines.length - 1))]);
  for (const k of AX) {
    const x = v[k];
    if (x === undefined) { $('v_'+k).textContent = '–'; continue; }
    $('v_'+k).textContent = x.toFixed(3);
    $('b_'+k).style.width = (x * 100).toFixed(1) + '%';
  }
}
function meta(c) {
  // 显示**去偏后**的保留率。原始 OLS 比值会被回归稀释压低（实测把 67% 报成 28%），
  // 页面显示错的数字会直接误导判断。
  const cl = (c.k_rot < 0.75 || c.k_tra < 0.9) ? 'warn' : '';
  $('meta').innerHTML = `<b>${c.name}</b><br>` +
    `<span class="muted">${c.lines.length} 帧 @ ${c.fps.toFixed(2)}fps · ` +
    `${(c.lines.length/c.fps).toFixed(1)}s</span><br>` +
    `<span class="${cl}">幅度保留率（已去偏）：旋转 ${c.k_rot.toFixed(2)} · ` +
    `平移 ${c.k_tra.toFixed(2)}　1.0 = 完整保留</span>`;
}
const vid = $('vid');
function loadClip(i) {
  clip = i;
  const c = CLIPS[i];
  lines = c.lines; fps = c.fps; idx = 0;
  $('vsel').innerHTML = c.videos.map(v => `<option value="${v.file}">${v.label}</option>`).join('');
  if (c.videos.length) { vid.src = c.videos[0].file; }
  meta(c);
  paint();
}
// 同步：视频时间是主，模拟器跟随
function advanceTo(target) {
  if (target > idx) { for (; idx < target && idx < lines.length; idx++) write(lines[idx]); }
  else if (target < idx - 1) {                       // 模拟器不能重置状态，回退就从头快进
    idx = 0;
    for (; idx < target && idx < lines.length; idx++) write(lines[idx]);
  }
}
function tick() {
  if (lines.length && vid.duration && !vid.paused) {
    advanceTo(Math.min(lines.length - 1, Math.floor(vid.currentTime * fps)));
  }
  paint();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

$('csel').onchange = e => { vid.pause(); loadClip(+e.target.value); };
$('vsel').onchange = e => {
  const t = vid.currentTime;
  vid.src = e.target.value; idx = 0;
  vid.addEventListener('loadedmetadata', () => { vid.currentTime = t; }, { once: true });
};
$('pp').onclick = () => { vid.paused ? vid.play() : vid.pause(); };
vid.onplay = () => { $('pp').textContent = '暂停'; };
vid.onpause = () => { $('pp').textContent = '播放'; };
$('rs').onclick = () => { idx = 0; vid.currentTime = 0; vid.play(); };
$('sp').oninput = e => {
  const r = +e.target.value;
  $('spv').textContent = r.toFixed(1) + 'x';
  vid.playbackRate = r;
};
loadClip(0);
</script>
</body>
</html>
"""


def read_clip(tc: Path, name: str, videos: list[dict], gain: tuple) -> dict:
    """tc.name 形如 `X_1280p_30fps.tcode.txt`。注意 Path.stem 只去掉最后一个后缀，
    会得到 `X_1280p_30fps.tcode`——必须自己剥掉完整后缀。"""
    lines = [s.strip() for s in tc.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not lines:
        raise ValueError(f"{tc} 是空的")
    m = re.search(r"I(\d+)", lines[0])
    iv = int(m.group(1)) if m else 33
    return {"name": name, "lines": lines, "fps": float(round(1000.0 / iv)),
            "videos": videos, "k_rot": gain[0], "k_tra": gain[1]}


def retention(clip_dir: Path) -> tuple[float, float]:
    """从同素材的 incam npz 现算**去偏后**的幅度保留率 (旋转, 平移)。

    这里必须用去偏值，不能用 OLS 斜率——自变量（关键点）带噪声会把斜率系统性压低
    （回归稀释），实测把 67% 的保留率报成 28%。页面上显示错的数字会误导判断。
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from viz import sources
        import to_funscript
        npzs = list(clip_dir.glob("gvhmr_rtmpose/*/hmr4d_results_incam.npz"))
        if not npzs:
            return float("nan"), float("nan")
        s = sources.from_smplx_npz(str(npzs[0]), fps=30.0)
        return to_funscript.estimate_retention(s)
    except Exception:
        return float("nan"), float("nan")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="生成 osr-emu 同步对比页（可含全部素材）")
    ap.add_argument("tcode", nargs="?", help=f"{TC_SUFFIX} 文件（省略则配合 --scan）")
    ap.add_argument("--scan", action="store_true",
                    help="扫描 workspace/funscripts 下全部素材，自动配对视频")
    ap.add_argument("--out", required=True, help="输出目录（html / 库 / 视频都放这里）")
    ap.add_argument("--video", action="append", default=[], help="单素材模式下附加的视频")
    ap.add_argument("--lib", default=str(LIB_DEFAULT), help="osr-emu.js 路径")
    ap.add_argument("--name", default=None, help="单素材模式下的显示名")
    ap.add_argument("--no-gain", action="store_true", help="跳过扭胯保真度计算（省时间）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = Path(args.lib).resolve()
    if not lib.exists():
        print(f"找不到库 {lib}\n先执行：curl -sSL https://unpkg.com/osr-emu -o {lib}",
              file=sys.stderr)
        return 1
    dst_lib = out / "osr-emu.js"
    if not (dst_lib.exists() and dst_lib.resolve() == lib.resolve()):
        shutil.copy2(lib, dst_lib)

    clips = []
    if args.scan:
        for tc in sorted((ROOT / "workspace" / "funscripts").glob(f"*/*{TC_SUFFIX}")):
            clip_id = tc.parent.name
            stem = tc.name[: -len(TC_SUFFIX)]
            vids = []
            # 画面候选：原片（720x1280，看动作最清楚）与 quad（含叠加/正面/侧面）
            cand = [(ROOT / "workspace" / "processed" / f"{stem}.mp4", "原片"),
                    (ROOT / "workspace" / "outputs" / clip_id / "quad.mp4", "quad（原片|叠加|正面|侧面）")]
            for p, label in cand:
                if not p.exists():
                    continue
                # **必须加素材名前缀**：5 条素材的 quad 都叫 quad.mp4，直接按原名复制
                # 到同一目录会互相覆盖，页面上所有素材就都指向最后一个了。
                dst = out / f"{clip_id}__{p.name}"
                if not (dst.exists() and dst.resolve() == p.resolve()):
                    shutil.copy2(p, dst)
                vids.append({"file": dst.name, "label": label})
            if not vids:
                print(f"⚠ {clip_id} 没有可用视频，跳过", file=sys.stderr)
                continue
            g = (float("nan"), float("nan")) if args.no_gain else \
                retention(ROOT / "workspace" / "outputs" / clip_id)
            clips.append(read_clip(tc, clip_id, vids, g))
    else:
        if not args.tcode:
            print("要么给 tcode 文件，要么加 --scan", file=sys.stderr)
            return 1
        tc = Path(args.tcode).resolve()
        if not tc.exists():
            print(f"找不到 {tc}", file=sys.stderr)
            return 1
        vids = []
        for v in args.video:
            p = Path(v).resolve()
            if not p.exists():
                print(f"⚠ 跳过不存在的视频 {p}", file=sys.stderr); continue
            dst = out / p.name
            if not (dst.exists() and dst.resolve() == p.resolve()):
                shutil.copy2(p, dst)
            vids.append({"file": p.name, "label": p.stem})
        name = args.name or tc.name[: -len(TC_SUFFIX)]
        g = (float("nan"), float("nan")) if args.no_gain else \
            retention(ROOT / "workspace" / "outputs" / tc.parent.name)
        clips.append(read_clip(tc, name, vids, g))

    if not clips:
        print("没有可用的素材", file=sys.stderr)
        return 1

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", "replay-all").strip("_") or "replay"
    page = out / f"{slug}.emu.html"
    page.write_text(PAGE.replace("__CLIPS__", json.dumps(clips, ensure_ascii=False)),
                    encoding="utf-8")

    print(f"已写出 {page}")
    print(f"  {len(clips)} 条素材：")
    for c in clips:
        print(f"    {c['name']:16s} {len(c['lines']):5d} 帧 @ {c['fps']:g}fps   "
              f"保留率 旋转{c['k_rot']:.2f} 平移{c['k_tra']:.2f}   "
              f"画面 {[v['label'] for v in c['videos']]}")
    print(f"\n**必须走 HTTP**（file:// 下浏览器以 CORS 拒绝加载 ES 模块，页面会空白）：")
    print(f'  python -m http.server 8765 --directory "{out}"')
    print(f"  打开 http://localhost:8765/{page.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
