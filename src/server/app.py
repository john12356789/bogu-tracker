"""骨骼追踪 Web 后端 —— Flask，默认 5000 端口（沿用 FSP 前端的硬编码约定）。

视频一律走 send_file(conditional=True) 的 Range 流式传输。
不能把 mp4 整个读进内存再吐出去：原片最大 671 MB，而列表页一条都不用下载，
只有真正点开播放的那条、且只下载播放到的那一段。

坐标一致性提醒：`keypoints_2d` 是在**预处理产物**（`workspace/processed/*.mp4`）
的像素空间里算的，不是原片。所以默认播预处理产物；原片只在需要音轨时用。
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from . import (
    cleanup, convert, device, index, jobs, motion, paths, presets,
    previews, thumbs,
)

import scene_cuts  # noqa: E402  —— 必须在 paths 之后：它负责把 src/ 放进 sys.path
import tracklet_merge  # noqa: E402

app = Flask(__name__, static_folder=None)
CORS(app)


@app.errorhandler(HTTPException)
def _json_error(exc: HTTPException):
    """让 abort(404, description=...) 返回 JSON 而不是 Flask 的 HTML 错误页。

    前端只解析 JSON；不处理的话错误信息会变成一坨 html，用户在界面上看到的
    就是「<!doctype html>...」而不是「没有这条素材：xxx」。
    """
    return jsonify({"error": exc.description, "code": exc.code}), exc.code


@app.errorhandler(Exception)
def _json_unhandled(exc: Exception):
    """**没预料到的异常也返回 JSON。**

    不加这条的话，任何漏网的异常（比如删视频时 Windows 的文件占用）都会走
    Flask 默认的 HTML 错误页，而前端只解析 JSON——于是界面上显示的是
    "The server encountered an internal error and was unable to complete your
    request"，一句有用的信息都没有，用户连"是什么操作失败"都看不出来。

    真实异常类型和消息照样带出去：这是本地工具，没有泄露面，而它正是排查
    唯一的线索。堆栈照常进后端日志。
    """
    if isinstance(exc, HTTPException):
        return _json_error(exc)
    app.logger.exception("未处理的异常")
    return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


# --------------------------------------------------------------------------
# 素材索引
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "clips": len(index.scan())})


@app.get("/api/clips")
def list_clips():
    """素材列表。只返回索引，不碰视频字节。"""
    return jsonify({"clips": [c.to_json() for c in index.scan()]})


@app.get("/api/clips/<slug>")
def get_clip(slug: str):
    clip = index.by_slug(slug)
    if clip is None:
        abort(404, description=f"没有这条素材：{slug}")
    return jsonify(clip.to_json())


# --------------------------------------------------------------------------
# 视频流
# --------------------------------------------------------------------------
def _video_for(slug: str, source: str) -> Path | None:
    if source == "original":
        clip = index.by_slug(slug)
        return (paths.VIDEOS / clip.filename) if clip else None
    return paths.processed_video(slug)


@app.get("/api/clips/<slug>/video")
def clip_video(slug: str):
    """Range 流式。默认给预处理产物（与 2D 关键点同像素空间），
    ?source=original 给原片（有音轨）。"""
    src = request.args.get("source", "processed")
    path = _video_for(slug, src)
    # 预处理产物缺失时退回原片，总比 404 强
    if path is None and src == "processed":
        path = _video_for(slug, "original")
    if path is None or not path.exists():
        abort(404, description=f"{slug} 没有可播放的视频（source={src}）")
    return send_file(path, conditional=True, mimetype="video/mp4")


@app.get("/api/clips/<slug>/quad")
def clip_quad(slug: str):
    """四格预览：原片 | 叠加 | mesh正面 | mesh侧面。"""
    p = paths.quad_path(slug)
    if p is None:
        abort(404, description=f"{slug} 没有 quad.mp4")
    return send_file(p, conditional=True, mimetype="video/mp4")


@app.get("/api/clips/<slug>/thumb")
def clip_thumb(slug: str):
    """素材缩略图（抽一帧）。第一次请求时现抽并落盘，之后是纯读文件。

    抽不出来就 404——**列表页少一张图，不该让整个页面报错**。
    """
    clip = index.by_slug(slug)          # 只查一次：列表是懒扫描，重复查不划算
    if clip is None:
        abort(404, description=f"没有这条素材：{slug}")
    p = thumbs.ensure(slug, clip.duration)
    if p is None or not p.exists():
        abort(404, description=f"{slug} 抽不出缩略图")
    # 缩略图对一条素材是不变的，让浏览器也缓存住，滚动来回时不再打服务端
    return send_file(p, mimetype="image/jpeg", max_age=3600)


# --------------------------------------------------------------------------
# 多人：轨迹段
# --------------------------------------------------------------------------
def _groups_of(slug: str) -> list[dict]:
    """这条素材**当前生效**的分组。**归并默认关闭**——只有在「人物」面板上
    点过"用这组"（写进 people_merge.json）的才算（见 tracklet_merge.effective）。"""
    return tracklet_merge.effective(tracklet_merge.load(paths.merge_json(slug)))


def _group_built(slug: str, tids: list[int]) -> bool:
    """这一组是不是**按这个分组**跑过（batch_run 的归并步骤留下的凭证）。

    凭证对不上（分组改过 / 换过预处理产物）就算没跑过——那正是要重跑的情形。
    """
    root = paths.rt_root(slug, tids[0])
    if root is None:
        return False
    try:
        return json.loads((root / "merge_spec.json").read_text(encoding="utf-8")) == sorted(tids)
    except (OSError, ValueError):
        return False


def _rows(slug: str, info: dict, groups: list[dict] | None = None) -> list[dict]:
    """这条素材现在有哪几个"人"——**已归并的组只占一行**。

    为什么必须合：确认归并之后，组里那几段就不是"可选的人"了，它们是同一个人的
    碎片。列表里留着它们，等于让人还能选到"半个人"，而且"选谁算脚本"这件事在
    两行上会给出**不同的答案**（选碎片 = 用那条碎片的重建结果，缺的帧靠插值）。

    行的 tid = 组内最小 tid（= 产出那一份 npz 的目录名），`tids` 是被合进来的
    全部轨迹号（界面拿它显示"人 1 = 1 + 11 + 14"）。没被合过的段原样返回
    （`tids` 只有它自己）。
    """
    segs = info.get("segments") or []
    if groups is None:
        groups = _groups_of(slug)
    # 噪声轨（dropped）也要进索引：它们可能被并进了某个组，那一行要算上它们
    # 的帧和框。
    by_tid = {int(s["tid"]): s for s in
              list(segs) + list(info.get("dropped") or [])}
    rows: list[dict] = []
    used: set[int] = set()
    for g in groups:
        members = [by_tid[t] for t in g["tids"] if t in by_tid]
        if len(members) < 2:                      # 只剩一段了：没得合
            continue
        used.update(int(m["tid"]) for m in members)
        frames = sorted({int(f) for m in members for f in (m.get("frames") or [])})
        # 组内区间两两不相交（见 tracklet_merge 的硬约束），所以下面按帧号拼
        # 不会出现两个框抢同一帧。
        pairs = sorted(((int(f), b) for m in members
                        for f, b in zip(m.get("frames") or [], m.get("boxes") or [])),
                       key=lambda fb: fb[0])
        runs, start = [], frames[0]
        for i in range(1, len(frames)):
            if frames[i] != frames[i - 1] + 1:
                runs.append([start, frames[i - 1]])
                start = frames[i]
        runs.append([start, frames[-1]])
        bi = max(range(len(pairs)), key=lambda i: (pairs[i][1][2] - pairs[i][1][0])
                 * (pairs[i][1][3] - pairs[i][1][1]))
        rows.append({
            "tid": min(int(m["tid"]) for m in members),
            "tids": sorted(int(m["tid"]) for m in members),
            "coverage": round(len(frames) / info["n_frames"], 4) if info.get("n_frames") else None,
            "area_sum": round(sum(float(m.get("area_sum") or 0) for m in members), 4),
            "runs": runs, "frames": frames, "boxes": [b for _, b in pairs],
            "thumb_frame": pairs[bi][0], "thumb_box": pairs[bi][1],
        })
    for s in segs:
        if int(s["tid"]) in used:
            continue
        rows.append({**s, "tids": [int(s["tid"])]})
    # 原顺序是"累计占画面面积"从大到小（prep_boxes 定的），合完保持一致
    rows.sort(key=lambda r: -(r.get("area_sum") or 0))
    return rows


@app.get("/api/clips/<slug>/people")
def clip_people(slug: str):
    """这条素材跟出了哪几条轨迹段、各占多长、以及每段的代表帧小图。

    **一个人 ≠ 一条轨迹**：实测（5192 帧的双人素材）YOLO 跟出 6 条，两个人各被切成
    9~12 段（互相遮挡时反复丢失重捕），还在第 3244 帧附近整套换了号，外加 2 条
    0.3% 的噪声。所以这里返回的是**段**，界面负责把段归成"人"。
    """
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    p = paths.people_json(slug)
    # 人物时间线（哪一段用哪条轨迹）**由后台保存、在这里带出去**：它描述的是
    # "界面上那套脚本是怎么拼出来的"，不跟着 bbx_all.json 走（那是检测结果），
    # 也不该塞进 funscript（那是播放器读的）。前端拿它把时间轴的人物带画回原样，
    # 否则重新打开素材会显示成"整条一个人"——而脚本其实是拼出来的。
    timeline = convert.saved_person_timeline(slug)
    # 按哪个坐标系判"有没有重建结果"：前端切坐标系时这里要跟着走，
    # 否则会出现"面板说能用、重算却报没有那份 npz"。
    SPACE_UI = request.args.get("space", "incam")
    if p is None:
        return jsonify({"ok": True, "n_frames": None, "segments": [], "dropped": [],
                        "multi": False, "timeline": timeline})
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        abort(500, description=f"轨迹清单读不了：{exc}")

    merges = tracklet_merge.load(paths.merge_json(slug))
    eff = _groups_of(slug)
    segs = []
    for seg in _rows(slug, info, eff):
        tid = int(seg["tid"])
        is_merged = len(seg.get("tids") or []) > 1
        segs.append({
            "tid": tid,
            # 合过的行给全部轨迹号，界面才好写「人 1 = 1 + 11 + 14」；
            # 没合过的就是它自己，界面照旧只显示「轨迹 N」。
            "tids": seg.get("tids") or [tid],
            "merged": is_merged,
            "coverage": seg.get("coverage"),
            # 这一行**有没有能用的重建结果**。两种判据不一样，是有意的：
            #   合过的行 → 必须**按这个分组**跑过（分组凭证在）。光看 npz 会把
            #     「多人轨迹」按单条 tid 跑出来的那份当成合过的产物——那正是
            #     "以为已经合成、实际还是半个人"的情形。
            #   没合过的 → 老判据（有没有那份 npz）。
            # 多人的时候「多人轨迹」默认最多跑 3 条（见 batch_run.STEP_OPTS），
            # 所以第 4 个人往往只有检测框、没有重建——面板得说出来，否则用户
            # 选了它点重算，报的是"轨迹 N 还没有…重建结果"，而那句话离"你选了
            # 个没跑过的人"很远。
            "has_npz": (_group_built(slug, seg["tids"]) if is_merged
                        else paths.npz_for_track(slug, SPACE_UI, tid) is not None),
            "area_sum": seg.get("area_sum"), "runs": seg.get("runs") or [],
            "n_frames": len(seg.get("frames") or []),
            "thumb_frame": seg.get("thumb_frame") or 0,
            "thumb_box": seg.get("thumb_box"),
        })
    # （曾经有过一个 merged_into 字段，给"这条并进了哪个人"用。现在合过的段
    # 已经**不再单独列出来**了，那个字段就没有意义了——留着只会让界面在一行
    # 「人 3」上再挂一个「并入人 3」。）
    # 归并建议：**纯算，不花 GPU**（只读上面那份检测框，见 tracklet_merge）。
    # 所以它随 /people 一起给出去，界面上一打开就能看到"这几段是同一个人"。
    # **建议不自动生效**：生效的只有 merges（人点过的），建议全部列出来等人点头。
    # 已经点过的那几组从建议里剔掉——已经这样了还劝一次，用户会以为没生效。
    all_segs = list(info.get("segments") or []) + list(info.get("dropped") or [])
    try:
        groups = tracklet_merge.suggest(all_segs, info.get("n_frames"))
    except Exception as exc:                              # noqa: BLE001
        groups = []
        print(f"[people] 归并建议算不了（不影响其余字段）：{exc}", file=sys.stderr)
    done = {tuple(g["tids"]) for g in eff}
    suggest = [g for g in groups
               if len(g["tids"]) > 1 and tuple(g["tids"]) not in done]
    # 这一组**按这个分组跑过没有**。判据是那一根里的分组凭证（batch_run 的归并
    # 步骤写的），不是 npz——npz 和「多人轨迹」按单条 tid 跑出来的那份长得一样。
    # 界面靠它把按钮从"开始归并"换成"重新跑一遍"。
    for g in eff:
        g["built"] = _group_built(slug, g["tids"])
    return jsonify({
        "ok": True, "n_frames": info.get("n_frames"),
        "width": info.get("width"), "height": info.get("height"),
        "min_coverage": info.get("min_coverage"),
        "segments": segs, "dropped": info.get("dropped") or [],
        # 界面上那个「按时段指定人物」的开关只在多人时才有意义
        "multi": len(segs) > 1,
        # 保存着的人物时间线（哪一段用哪条轨迹）。前端拿它把时间轴的人物带画回
        # 原样——没有它，重新打开素材会显示成"整条一个人"，而脚本其实拼过。
        "timeline": timeline,
        "suggest": suggest,
        "merges": eff,
        # 曾经这里有一个 `auto`：生效的分组是算法自动给的那一份。归并改成默认关
        # 之后那个状态不存在了（生效的只有人点过的），字段一并去掉。
        # 「有没有决定过」（people_merge.json 在不在）也**不往上报**：它在"生效
        # 什么"上和"没有文件"完全一样，界面没有需要分开措辞的地方。
    })


@app.post("/api/clips/<slug>/people/merge")
def clip_people_merge(slug: str):
    """写分组决策。**只写决策，不跑管线**——跑由「人物」面板那两个按钮提交作业。

    收 `{"groups": [[2, 8, 15], [5, 9]]}`。**给空列表 = 撤销全部归并**（写一个空
    文件）。空文件和没有文件在"生效什么"上是同一件事（都不并，见
    tracklet_merge.effective），留着这个区别是因为界面要知道该说"还没确认过"
    还是"已取消"。
    """
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    body = request.get_json(force=True, silent=True) or {}
    raw = body.get("groups")
    if not isinstance(raw, list):
        abort(400, description="groups 得是一个数组的数组，例如 [[2, 8, 15]]")
    p = paths.people_json(slug)
    known = set()
    if p is not None:
        try:
            info0 = json.loads(p.read_text(encoding="utf-8"))
            # 噪声轨（dropped）也算已知——它们会被并进某个组，界面提交的组里
            # 就可能带着它们的 tid。
            known = {int(s["tid"]) for s in
                     list(info0.get("segments") or []) + list(info0.get("dropped") or [])}
        except (OSError, ValueError, KeyError, TypeError):
            known = set()
    groups = []
    for g in raw:
        try:
            tids = sorted({int(t) for t in g})
        except (TypeError, ValueError):
            abort(400, description=f"分组里出现了不是轨迹号的项：{g}")
        if len(tids) < 2:
            continue
        unknown = [t for t in tids if known and t not in known]
        if unknown:
            abort(400, description=f"这条素材没有轨迹 {unknown}（可能换了预处理产物）")
        groups.append({"gid": tids[0], "tids": tids})
    tracklet_merge.save(paths.merge_json(slug), groups)
    return jsonify({"ok": True, "groups": groups,
                    "file": str(paths.merge_json(slug).relative_to(paths.WORKSPACE))})


@app.get("/api/clips/<slug>/boxes")
def clip_boxes(slug: str):
    """每条轨迹段的逐帧框（稀疏：只给有检出的帧）。

    画面上叠加"这是谁"、以及**点一下选人**都要它。只给框不给关键点：关键点只有
    跑过 rtmpose 的轨迹才有，而"选人"这件事在**跑之前**就得能做（否则用户没有
    依据去选）。框是按预处理产物的像素算的，前端拿 width/height 归一化。
    """
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    p = paths.people_json(slug)
    if p is None:
        return jsonify({"ok": True, "n_frames": None, "segments": []})
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        abort(500, description=f"轨迹清单读不了：{exc}")
    # **和 /people 用同一套行**：合过的那几段在画面上也要算成一个"人"，
    # 否则点一下画面选中了 tid 14，而列表里根本没有这一行（它已经并进人 1）。
    return jsonify({
        "ok": True, "n_frames": info.get("n_frames"),
        "width": info.get("width"), "height": info.get("height"),
        "segments": [{"tid": r["tid"], "tids": r.get("tids") or [r["tid"]],
                      "frames": r.get("frames") or [],
                      "boxes": r.get("boxes") or []}
                     for r in _rows(slug, info)],
    })


@app.get("/api/clips/<slug>/segment/<int:tid>/thumb")
def clip_segment_thumb(slug: str, tid: int):
    """某条轨迹段的代表图（从框里裁）。列表里用来认人。"""
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    p = paths.people_json(slug)
    if p is None:
        abort(404, description=f"{slug} 没有轨迹清单")
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        abort(404, description="轨迹清单读不了")
    # 走 /people 那套行：合过的行 tid = 组内最小 tid，代表帧要**在整个组里**挑
    # （框最大的那帧），只用碎片自己的帧挑出来的图可能又小又偏。
    for r in _rows(slug, info):
        if int(r["tid"]) != tid:
            continue
        frame, box = r.get("thumb_frame"), r.get("thumb_box")
        if not frame or not box:
            break
        img = thumbs.segment_thumb(slug, tid, int(frame), list(box))
        if img is None:
            abort(404, description=f"track {tid} 的图裁不出来")
        return send_file(img, mimetype="image/jpeg", max_age=3600)
    abort(404, description=f"没有 track {tid}")


@app.get("/api/clips/<slug>/axes")
def clip_axes(slug: str):
    """这条素材**现在**有哪些轴。

    给"点开一条素材就加载它的脚本"用。轴清单必须现问，不能用 /api/clips
    列表里的快照：列表是缓存的，而 `提交`（promote）只往盘上写、不重扫索引，
    照快照取会一个轴都取不到——现象就是"文件明明在 workspace/funscripts 里，
    时间轴却是空的"。这个端点直接读磁盘，绕开这层缓存。
    """
    return jsonify({"axes": list(paths.funscript_files(slug))})


# --------------------------------------------------------------------------
# 切点
# --------------------------------------------------------------------------
# 检测整个进程串行：它要解码整条视频（几分钟的素材几秒，二十分钟的要几十秒），
# 并发跑几次只是互相抢 CPU，而且第一次的结果一落盘后面就全走缓存了。
# 加锁而不是加队列：调用方（打开素材的前端）要的就是"等一会儿拿到答案"，
# 不是"过会儿再来问"。
_CUTS_LOCK = threading.Lock()


def _cuts_payload(slug: str, data: dict | None) -> dict:
    if not data:
        return {"cuts": [], "detected": False, "n_frames": None, "fps": None}
    return {
        "cuts": data.get("cuts", []),
        "detected": True,
        "fps": data.get("fps"),
        "n_frames": data.get("n_frames"),
        # 阈值信息一并给出：分数分布因素材而异，"为什么这条一个切点都没有"
        # 得能从那几个数上看出来，而不是只能选择相信
        "threshold": data.get("threshold"),
        "median": data.get("median"),
        "sigma": data.get("sigma"),
        # 这份清单是**按哪套判据**测出来的。面板上改过判据之后，得知道盘上
        # 这份是旧的还是新的。
        "params": data.get("params"),
    }


@app.get("/api/cuts/params")
def cuts_params():
    """切点判据的默认值与取值域。默认值不在这里写死——它和 CLI、
    落盘的 params 共用 scene_cuts.TUNABLE 那一份定义。"""
    return jsonify({"defaults": scene_cuts.defaults(),
                    "limits": {k: list(v) for k, v in scene_cuts.LIMITS.items()}})


@app.get("/api/clips/<slug>/cuts")
def clip_cuts(slug: str):
    """切点清单。有缓存直接给，没有就**现测并落盘**。

    跑在预处理产物上（整条管线的帧空间以它为准），检测本身见 scene_cuts.py。
    检测不到切点很正常——大量素材本来就是单镜头，返回空清单不是错误。
    """
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    with _CUTS_LOCK:
        data = scene_cuts.load(paths.cuts_json(slug))     # 等锁的期间可能已经有人算完了
        if data is None:
            video = paths.processed_video(slug)
            if video is None:
                abort(400, description=f"{slug} 没有预处理产物，检测不了切点")
            try:
                data = scene_cuts.detect(video)
            except Exception as exc:                      # noqa: BLE001
                abort(500, description=f"切点检测失败：{exc}")
            scene_cuts.save(data, paths.cuts_json(slug))
    return jsonify(_cuts_payload(slug, data))


@app.post("/api/clips/<slug>/cuts")
def clip_cuts_redetect(slug: str):
    """重新检测（界面上那个按钮）。覆盖缓存。

    body 里可以带判据（abs_min / k / sust_min / prom_min / min_gap /
    change_level），缺的用默认值补、越界的夹回来——见 scene_cuts.normalize。
    判据**不落进 store**：它跟着这一次检测写进 cuts.json 的 params 里，
    下次点开直接读缓存，不会因为面板上滑杆被拖过就悄悄改变已有结果。
    """
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    video = paths.processed_video(slug)
    if video is None:
        abort(400, description=f"{slug} 没有预处理产物，检测不了切点")
    params = request.get_json(force=True, silent=True) or {}
    with _CUTS_LOCK:
        try:
            data = scene_cuts.detect(video, params)
        except Exception as exc:                          # noqa: BLE001
            abort(500, description=f"切点检测失败：{exc}")
        scene_cuts.save(data, paths.cuts_json(slug))
    return jsonify(_cuts_payload(slug, data))


@app.get("/api/clips/<slug>/funscript/<axis>")
def clip_funscript(slug: str, axis: str):
    files = paths.funscript_files(slug)
    p = files.get(axis)
    if p is None:
        abort(404, description=f"{slug} 没有 {axis} 轴的 funscript")
    return send_file(p, mimetype="application/json")


# --------------------------------------------------------------------------
# 运动数据（关节 + 2D 关键点）
# --------------------------------------------------------------------------
@app.get("/api/clips/<slug>/tcode")
def clip_tcode(slug: str):
    """TCode 指令流——osr-emu 的输入。来自正式的 workspace/funscripts/<slug>/。

    参数重算后用的不是这个端点：/convert 的返回值里直接带新的 tcode，
    这样模拟器能跟着参数立刻变。

    这里**不传 fps**（正式产物旁边没有记录真实帧率的东西），走 read_tcode 的
    取整回落。这条管线产出的 tcode 一律是 30fps（to_funscript 的 `I` =
    round(1000/fps)，写出来是 `I33`），取整恰好还原成 30，是精确的。
    换过帧率的产物要在这里显式传 fps——但从 `I33` 反推的 30.303fps 会
    在长素材上累积出好几秒的偏移，别再用它了。
    """
    d = paths.funscript_dir(slug)
    tc = convert.read_tcode(d) if d else None
    if tc is None:
        abort(404, description=f"{slug} 没有 tcode.txt")
    return jsonify(tc)


# --------------------------------------------------------------------------
# 设备：把脚本发给真实硬件（T-code v0.3）
# --------------------------------------------------------------------------
# 整个后端**只有一个** DeviceLink，和 `_JOBS` 一样是进程级的。理由见 device.py
# 文件头：一条管线上两个设备是以后的事。
_DEVICE = device.DeviceLink()


@app.get("/api/device")
def device_status():
    """连接状态。前端连接期间轮询它画那个点（断线由设备的线程自己发现）。"""
    return jsonify(_DEVICE.status())


@app.get("/api/device/ports")
def device_ports():
    """串口清单。

    **必须放到线程里**：枚举端口在 pyserial 里是一次 WMI 调用，实测可以卡几百
    毫秒，而那段时间整个 Flask 请求线程是被占住的。前端是"要的时候才问一次"
    （插上设备再按刷新），不是每次渲染都问。
    """
    return jsonify({"ports": device.list_ports()})


@app.post("/api/device/open")
def device_open():
    """开始发送。**参数体就是设备配置**，字段见 device.DeviceLink.open。

    没有 `rate` / `interpolate`：这一版的节奏来自脚本自己（行尾的 `I<ms>`），
    理由见 device.py 文件头的"差异一/差异二"。
    """
    body = request.get_json(force=True, silent=True) or {}
    return jsonify(_DEVICE.open(body))


@app.post("/api/device/close")
def device_close():
    """停止发送，设备留在原地（不命令任何位置，理由见 device.py）。"""
    return jsonify(_DEVICE.close())


@app.post("/api/device/line")
def device_line():
    """喂**一条**要发给设备的 TCode 行。前端跟着播放头调它，30Hz 左右。

    这条路上什么都不能慢：`feed()` 只做一次赋值和两次置位（设备的写在自己的
    线程上）。返回值只有"收下没有"——**行是白名单校验过的**（见 device.valid_line），
    因为这一行最终会变成真实硬件的一次运动，而它来自请求体。
    """
    body = request.get_json(force=True, silent=True) or {}
    if not _DEVICE.feed(body.get("line") or ""):
        abort(400, description="不是合法的 TCode 行")
    return jsonify({"ok": True})


@app.get("/api/clips/<slug>/motion")
def clip_motion(slug: str):
    """逐帧 3D 关节 + 2D 关键点 + 拓扑定义。

    ?space=global（重力对齐、**+Y 朝上**、米制）或 ?space=incam
    （OpenCV 相机系、**+Y 朝下**，默认）。两者不是同一个世界系的两种视角。

    **3D 显示两种都能看**：客户端不假设坐标轴标签，而是由关节自己推出"上"
    （spine1-pelvis），incam 下点积为负就整体翻 180°（见 core/skeleton3d.ts 的
    detectFlip）。所以默认给 incam 不会让人倒过来。

    返回**已经序列化好的字符串**而不是 dict：Flask 的 jsonify 会把 5MB 的
    dict 再序列化一遍，那一段实测约 600ms，是切换素材卡顿的三分之一。
    """
    space = request.args.get("space", "global")
    # ?lite=1：只要元信息（帧数/帧率/画面尺寸），不要那 2.5 MB 逐帧数据。
    # 前端打开素材时走这条（时间轴要用 n_frames/fps，但默认没有任何视图需要逐帧），
    # 真打开 2D/3D 视图时再取完整的。
    lite = request.args.get("lite") in ("1", "true", "yes")
    text = motion.describe_json(slug, space, lite=lite)
    if text is None:
        d = motion.describe(slug, space, lite=lite)
        abort(404, description=d.get("error", "无数据"))
    return Response(text, mimetype="application/json")


# --------------------------------------------------------------------------
# 参数重算
# --------------------------------------------------------------------------
@app.get("/api/params")
def param_defaults():
    """to_funscript 的可调参数与默认值。默认值直接取自 argparse，不在这里写死。"""
    return jsonify({"defaults": convert.defaults()})


@app.get("/api/presets")
def presets_list():
    """全部参数预设，新的在前。

    返回的每一份都是**补全过的完整参数**（缺的键补默认值），前端拿到就能直接铺上去，
    不用再和 paramDefaults 合一次——合的逻辑只留一份，在 presets.normalize 里。
    """
    return jsonify({"presets": presets.list_all(), "dir": str(presets.DIR)})


@app.post("/api/presets")
def presets_save():
    """存一份预设（同名覆盖）。body: {name, params, cuts, space}。

    名字之外的内容逐键校验（见 presets 模块的说明）：预设是盘上的普通文件，
    可能被手改过、也可能来自旧版本，不能整份信任。
    """
    body = request.get_json(force=True, silent=True) or {}
    try:
        item = presets.save(body.get("name"), body)
    except ValueError as exc:
        abort(400, description=str(exc))
    except OSError as exc:
        abort(500, description=f"写预设失败：{exc}")
    return jsonify(item)


@app.delete("/api/presets/<name>")
def presets_delete(name: str):
    if not presets.delete(name):
        abort(404, description=f"没有这份预设：{name}")
    return jsonify({"ok": True, "name": name})


@app.post("/api/clips/<slug>/convert")
def clip_convert(slug: str):
    """按传入参数重跑一遍 funscript 生成。

    走的是 to_funscript.run() 本身——数学只有一份实现，后端不重写公式。
    seq 常驻缓存，所以这是一次 0.25 秒量级的操作，可以直接挂在滑杆上。
    产物写到 preview 目录，**不覆盖** workspace/funscripts 下的正式产物。
    """
    params = request.get_json(force=True, silent=True) or {}
    result = convert.convert(slug, params)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.post("/api/clips/<slug>/promote")
def clip_promote(slug: str):
    """把 preview 里调好的产物提交成正式产物（会覆盖 funscripts/<slug>/）。

    **不区分坐标系**：一条素材只有一份脚本。用哪个坐标系生成是「重算」那一步
    的事（`convert()` 里选 npz），提交只管把当前那份草稿落成正式产物——
    两者文件名本来就一样，本来就是互相覆盖的关系。
    """
    import shutil

    src = convert.preview_dir(slug)
    if not src.is_dir() or not paths.funscript_files_in(src):
        abort(404, description=f"{slug} 还没有调好的 preview 产物，先点「重算」")
    dst = paths.FUNSCRIPTS / slug
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(src.iterdir()):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            copied.append(p.name)
    # 正式目录里不属于这一套的旧文件删掉，**"唯一一份脚本"才是真的**：
    # 换过检测器/坐标系之后产物名前缀可能变（名字取自 npz 所在目录名），
    # 只复制不清理就会留下两套同名不同前缀的脚本，而接口是按后缀取第一个。
    stale = [p.name for p in sorted(dst.iterdir())
             if p.is_file() and p.name not in copied]
    for name in stale:
        (dst / name).unlink()
    return jsonify({"ok": True, "dir": str(dst), "copied": copied, "removed": stale})


# --------------------------------------------------------------------------
# GVHMR 作业队列（单并发：6 GB 显存只能跑一个）
# --------------------------------------------------------------------------
@app.get("/api/steps")
def step_list():
    """可选的管线步骤。清单与默认值都来自 batch_run，不在这里写死。"""
    return jsonify(jobs.steps_meta())


@app.get("/api/jobs")
def job_list():
    jobs.start()
    return jsonify({"jobs": [j.to_json() for j in jobs.all_jobs()]})


@app.post("/api/jobs")
def job_create():
    """body: {slug, steps?: [...], force?: bool, step_opts?: {步骤 id: {...}}}

    steps 省略时用 batch_run 的默认集合（前 5 步，不含「多人轨迹」和渲染）。
    step_opts 是各步骤自己的参数（「导出 npz」导哪个坐标系、「多人轨迹」的阈值和
    条数上限），没勾的那一步的参数不起作用。**不在后端再夹一次范围**——
    `jobs.submit` 里调的就是 batch_run 那个 normalize，和命令行走同一条路。
    """
    body = request.get_json(force=True, silent=True) or {}
    slug = body.get("slug")
    clip = index.by_slug(slug) if slug else None
    if clip is None:
        abort(404, description=f"没有这条素材：{slug}")

    steps = jobs.normalize_steps(body.get("steps"))
    if steps is None:
        abort(400, description=f"步骤不合法，可选：{', '.join(jobs.steps_meta()['order'])}")
    # 依赖检查放在这里而不是只放界面上：界面不是唯一入口，curl 直接提交一样
    # 能绕过。放行的话失败发生在跑了几分钟之后（导出 npz 那步），只留下一句
    # 文件不存在的报错，看不出是"前置没跑过"。
    why = jobs.validate(clip.slug, steps)
    if why:
        abort(400, description=why)
    job = jobs.submit(clip.slug, clip.stem, steps, force=bool(body.get("force")),
                      step_opts=body.get("step_opts"))
    return jsonify(job.to_json()), 202


# --------------------------------------------------------------------------
# 删除素材（产物 / 可选连原片）
# --------------------------------------------------------------------------
@app.get("/api/clips/<slug>/usage")
def clip_usage(slug: str):
    """这条素材占了多少盘，分门别类——删除确认弹窗里显示的清单。"""
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    return jsonify(cleanup.usage(slug))


@app.delete("/api/clips/<slug>")
def clip_delete(slug: str):
    """删掉产物；`?with_video=1` 连原片一起删。

    **不可逆**，所以界面那边必须有确认弹窗，而且原片默认不删。这里再挡两道：
    名字白名单（slug 进 URL，`..` 能跑出工作区）、有作业在跑就拒绝。
    """
    if index.by_slug(slug) is None:
        abort(404, description=f"没有这条素材：{slug}")
    with_video = request.args.get("with_video") in ("1", "true", "yes")
    try:
        return jsonify(cleanup.delete(slug, with_video))
    except ValueError as exc:                     # 名字不合法
        abort(400, description=str(exc))
    except RuntimeError as exc:                   # 有作业在跑
        abort(409, description=str(exc))


# --------------------------------------------------------------------------
# 渲染预览（自己拼格子的对照视频，落 outputs/<slug>/，与 quad.mp4 并列，不覆盖它）
# --------------------------------------------------------------------------
@app.get("/api/clips/<slug>/previews")
def preview_list(slug: str):
    return jsonify({"previews": previews.list_previews(slug),
                    "default_views": previews.DEFAULT_VIEWS})


@app.post("/api/clips/<slug>/render_preview")
def preview_render(slug: str):
    """body: {views, grid?, cell_h?, frames?} —— 参数含义见 previews.normalize。

    渲染几分钟起步，所以走作业队列（和管线同一个队列：渲染吃 CPU/内存，
    GVHMR 吃显存，两个同时跑对这台机器都不好受）。
    """
    clip = index.by_slug(slug)
    if clip is None:
        abort(404, description=f"没有这条素材：{slug}")
    if not clip.has_results:
        abort(400, description="这条素材还没有重建结果，渲染不了预览")

    params, err = previews.normalize(request.get_json(force=True, silent=True) or {})
    if err:
        abort(400, description=err)

    out = previews.output_dir(clip.slug) / previews.make_name(params)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 元数据等渲染成功再写（见 previews.write_meta 的说明：失败/中断不该留下
    # 一条"看起来正常"的记录）
    job = jobs.submit_render(clip.slug, clip.stem, params, str(out))
    return jsonify({"job": job.to_json(), "preview": out.name,
                    "start_ms": previews._start_frame(params["frames"]) / previews.FPS * 1000}), 202


@app.get("/api/clips/<slug>/preview/<name>")
def preview_get(slug: str, name: str):
    p = previews.path_of(slug, name)
    if p is None:
        abort(404, description=f"没有这条预览：{name}")
    return send_file(p, mimetype="video/mp4", conditional=True)


@app.delete("/api/clips/<slug>/preview/<name>")
def preview_delete(slug: str, name: str):
    if not previews.delete(slug, name):
        abort(404, description=f"没有这条预览：{name}")
    return jsonify({"ok": True})


@app.get("/api/jobs/<int:job_id>")
def job_get(job_id: int):
    job = jobs.get(job_id)
    if job is None:
        abort(404, description=f"没有这个作业：{job_id}")
    return jsonify(job.to_json())


@app.delete("/api/jobs/<int:job_id>")
def job_cancel(job_id: int):
    if not jobs.cancel(job_id):
        abort(409, description="作业不存在或已经结束")
    return jsonify(jobs.get(job_id).to_json())


def main() -> int:
    ap = argparse.ArgumentParser(description="骨骼追踪 Web 后端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"[server] 项目根 {paths.ROOT}")
    print(f"[server] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
