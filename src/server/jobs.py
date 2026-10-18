"""GVHMR 批处理作业队列。

**为什么是子进程而不是线程**：进度只能从 batch_run 的输出里拿。在线程里做
就得劫持全局 sys.stdout，而 Flask 是多线程的——别的请求的打印会被一起抓进来。
起独立进程则各走各的管道，还能干净地杀掉。

**为什么并发度固定为 1**：6 GB 显存是硬约束，GVHMR 的 HMR2 特征提取一批就占约 5 GB。
两个作业并行必然 OOM。队列天然串行，不提供并发度参数。

进度由 batch_run 自己在 stdout 上发结构化事件（`@@EVENT {json}`），
这里只负责解析。**不让 server 去认 tqdm 和渲染的文本格式**——那些格式
只有 batch_run 知道，而且会变。

步骤 id / 名称 / 默认集合都从 batch_run import，不在这里重写一份。
"""

from __future__ import annotations

import itertools
import json
import queue
import subprocess
import sys
from pathlib import Path
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from . import paths  # 先把 src/ 放进 sys.path

from batch_run import (  # noqa: E402
    DEFAULT_STEPS, EVENT_PREFIX, HIDDEN_STEPS, PREREQS, STEP_NAME, STEP_OPTS, STEP_ORDER,
    check_prereqs, normalize_all_step_opts,
)

_JOBS: dict[int, "Job"] = {}
_ORDER: list[int] = []
_LOCK = threading.Lock()
_IDS = itertools.count(1)
_QUEUE: "queue.Queue[int]" = queue.Queue()
_WORKER: threading.Thread | None = None
_PROCS: dict[int, subprocess.Popen] = {}


# 界面只列**主线那五步**，另外两步在 batch_run 里保留但不摆出来：
#
#   render  它的位置被「渲染预览」取代了（能挑格子、挑时间段、产物落临时目录）。
#           命令行 `--steps ...,render` 还能补跑正式 quad.mp4。
#   merge   「归并人物」的入口在**「人物」面板**：那一步要先确认分组才有事可做，
#           摆在勾选面板里只会让人先勾上再跑、然后拿到一句"没有分组决策"
#           （真发生过）。面板里那个按钮提交的作业就是 `steps: ["merge"]`。
#
# 两步仍然在 STEP_ORDER/STEP_DEPS 里，依赖图和校验照旧认它们——这里只是
# "不再摆在界面上让人点"。
UI_STEPS = [s for s in STEP_ORDER
            if s != "render" and s not in HIDDEN_STEPS]


def steps_meta() -> dict:
    """给界面用的步骤清单与依赖图。

    prereqs 从 batch_run 直接带出去，界面**不另写一份**——依赖表在两个地方
    各存一份，迟早会漂移，而漂移的表现是"界面允许点、后端 400"。

    界面上（勾选面板、素材行底下那排圆点）都只用 order 里那几步。
    """
    return {
        "order": UI_STEPS,
        "names": STEP_NAME,
        "default": DEFAULT_STEPS,
        "prereqs": PREREQS,
        # 各步骤的可调参数（步骤 id -> 参数表）也直接带出去，同样不另写一份：
        # 界面对着它渲染输入框/选项，包括中文名、默认值、范围、说明，以及
        # "要不要显示"取决于对应那一步勾没勾。
        "stepOpts": STEP_OPTS,
    }


def normalize_steps(requested: list[str] | None) -> list[str] | None:
    """校验并规范化用户选的步骤。返回 None 表示输入非法。"""
    if requested is None:
        return list(DEFAULT_STEPS)
    unknown = [s for s in requested if s not in STEP_NAME]
    if unknown or not requested:
        return None
    return [s for s in STEP_ORDER if s in requested]   # 强制规范顺序


def validate(slug: str, steps: list[str]) -> str | None:
    """这套步骤在这条素材上能不能跑。None = 能，否则是不能的原因。

    判据要用**磁盘现状**（paths.step_products）：素材算过了就只勾后面几步，
    是常规用法，不能因为没勾前置就拦下来。规则本身在 batch_run.check_prereqs，
    纯函数，好测；这里只负责把产物查出来喂进去。
    """
    return check_prereqs(steps, paths.step_products(slug))


@dataclass
class Job:
    id: int
    slug: str
    stem: str                       # 原始文件名主干，用来给 --only 做精确匹配
    steps: list[str] = field(default_factory=list)
    force: bool = False
    # 各步骤的可调参数（步骤 id -> {参数: 值}）。没给就用 batch_run 的默认值——
    # 存的是**已经夹过范围**的值，界面和命令行走的是同一条校验。
    step_opts: dict = field(default_factory=dict)
    # 管线作业（跑 batch_run 的步骤）和渲染预览作业（临时渲一段对照视频）走同一个
    # 队列：渲染吃 CPU/内存，GVHMR 吃显存，两个一起跑对这台机器都不好受。
    kind: str = "pipeline"          # pipeline | render
    render: dict = field(default_factory=dict)   # kind=render 时的参数与产物路径
    result: dict = field(default_factory=dict)   # 跑完之后要给界面看的东西
    state: str = "queued"           # queued | running | done | failed | cancelled
    current: int = 0                # 当前步骤序号 1..N
    current_id: str = ""
    current_name: str = ""
    frac: float | None = None       # 当前步骤的子进度；None = 该步没有子进度
    step_states: dict = field(default_factory=dict)
    step_elapsed: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    log: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id, "slug": self.slug, "stem": self.stem,
            "steps": self.steps, "force": self.force, "state": self.state,
            "current": self.current, "current_id": self.current_id,
            "current_name": self.current_name, "frac": self.frac,
            "step_states": self.step_states, "step_elapsed": self.step_elapsed,
            "elapsed": (self.finished or time.time()) - (self.started or time.time())
            if self.started else 0.0,
            "created": self.created, "started": self.started,
            "finished": self.finished, "error": self.error,
            "kind": self.kind, "result": self.result,
            "log": self.log[-60:],
        }


def _push(job: Job, line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    if line.startswith(EVENT_PREFIX):
        try:
            ev = json.loads(line[len(EVENT_PREFIX):])
        except ValueError:
            return
        kind = ev.get("kind")
        if kind == "step":
            sid, state = ev.get("id", ""), ev.get("state")
            job.current = ev.get("index", job.current)
            job.current_id, job.current_name = sid, ev.get("name", "")
            job.step_states[sid] = state
            if state == "start":
                job.frac = None                    # 换步了，旧子进度作废
            elif state == "done":
                job.step_elapsed[sid] = ev.get("elapsed")
        elif kind == "progress":
            if ev.get("index") == job.current:
                job.frac = ev.get("frac")
        elif kind == "failed":
            job.error = ev.get("message")
        return
    job.log.append(line)
    if len(job.log) > 600:
        del job.log[:300]


def _run(job: Job) -> None:
    job.state = "running"
    job.started = time.time()
    job.step_states = {s: "pending" for s in job.steps}

    if job.kind == "render":
        cmd = [sys.executable, str(paths.SRC / "batch_run.py"),
               "--only", job.stem, "--render-preview",
               "--out", job.render["out"], "--views", job.render["views"]]
        for flag in ("grid", "cell_h", "frames"):
            if job.render.get(flag):
                cmd += [f"--{flag.replace('_', '-')}", str(job.render[flag])]
    else:
        cmd = [sys.executable, str(paths.SRC / "batch_run.py"),
               "--only", job.stem, "--steps", ",".join(job.steps)]
        # 不带 --force 的话，凡是已有产物的素材都会走"跳过"快路径，
        # 而 workspace 里的素材绝大多数都已经算过，那样队列就成了摆设。
        if job.force:
            cmd.append("--force")
        # 步骤参数**按 batch_run.STEP_OPTS 那张表拼**，不在这里再写一遍
        # "哪个步骤有哪个参数、命令行叫什么"——那种映射抄第二遍必定漂，
        # 而漂的表现是"界面上调了、跑的还是默认值"。
        # 只给**勾了的**那几步拼：没勾也拼的话日志里会多出用不上的参数，
        # 看不出这一步到底跑没跑。
        for sid in job.steps:
            for spec in STEP_OPTS.get(sid, []):
                v = (job.step_opts.get(sid) or {}).get(spec["key"])
                if v is not None:
                    cmd += [spec["flag"], str(v)]

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(paths.ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except OSError as exc:
        job.state, job.error, job.finished = "failed", str(exc), time.time()
        return

    with _LOCK:
        _PROCS[job.id] = proc

    tail: deque[str] = deque(maxlen=25)
    assert proc.stdout is not None
    for line in proc.stdout:
        _push(job, line)
        tail.append(line.rstrip())
    proc.wait()

    with _LOCK:
        _PROCS.pop(job.id, None)
    job.finished = time.time()

    if job.state == "cancelled":
        return
    if proc.returncode == 0 and job.error is None:
        job.state = "done"
        job.frac = 1.0
        if job.kind == "render":
            # 预览不进素材索引，界面靠 /api/clips/<slug>/previews 现扫目录。
            # **元数据在这里写**（不是提交时）：只有真跑出来了才算一条预览，
            # 不然中断的渲染会留下一条点开是空视频的记录。
            from . import previews
            previews.write_meta(Path(job.render["out"]), job.render)
            job.result = {"out": job.render["out"], "views": job.render["views"]}
        else:
            # 产物落盘了，缓存的 seq / 索引已过期，清掉让下次重读
            from . import convert, index, motion
            convert.clear(job.slug)
            motion.clear(job.slug)
            index.scan()      # 重扫一遍，把新产物写进缓存
    else:
        job.state = "failed"
        if not job.error:
            job.error = "\n".join(tail) or f"退出码 {proc.returncode}"


def _worker() -> None:
    while True:
        job_id = _QUEUE.get()
        job = _JOBS.get(job_id)
        if job is None:
            continue
        try:
            _run(job)
        except Exception as exc:                      # noqa: BLE001 — worker 绝不能死
            job.state, job.error, job.finished = "failed", str(exc), time.time()
        finally:
            _QUEUE.task_done()


def start() -> None:
    global _WORKER
    if _WORKER is not None and _WORKER.is_alive():
        return
    _WORKER = threading.Thread(target=_worker, name="gvhmr-worker", daemon=True)
    _WORKER.start()


def submit(slug: str, stem: str, steps: list[str], force: bool = False,
           step_opts: dict | None = None) -> Job:
    start()
    job = Job(id=next(_IDS), slug=slug, stem=stem, steps=steps, force=force,
              step_opts=normalize_all_step_opts(step_opts))
    with _LOCK:
        _JOBS[job.id] = job
        _ORDER.append(job.id)
    _QUEUE.put(job.id)
    return job


def submit_render(slug: str, stem: str, params: dict, out: str) -> Job:
    """提交一个渲染预览作业。params 来自 previews.normalize（已校验）。"""
    start()
    job = Job(id=next(_IDS), slug=slug, stem=stem, kind="render",
              steps=["render"], render={**params, "out": out})
    with _LOCK:
        _JOBS[job.id] = job
        _ORDER.append(job.id)
    _QUEUE.put(job.id)
    return job


def get(job_id: int) -> Job | None:
    return _JOBS.get(job_id)


def all_jobs() -> list[Job]:
    return [_JOBS[i] for i in _ORDER if i in _JOBS]


def cancel(job_id: int) -> bool:
    job = _JOBS.get(job_id)
    if job is None or job.state not in ("queued", "running"):
        return False
    job.state = "cancelled"
    job.finished = time.time()
    with _LOCK:
        proc = _PROCS.get(job_id)
    if proc is not None and proc.poll() is None:
        # Windows 上必须带 /T 杀整棵进程树：GVHMR 那步还会再起子进程
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    return True


def busy() -> bool:
    return any(j.state in ("queued", "running") for j in _JOBS.values())
