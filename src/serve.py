"""Web 后端入口。

为什么要多这一层：把 src/server/ 当包导入（而不是直接跑 app.py），
里面的相对导入（from . import paths）才成立。同时也保证了无论从哪个
工作目录调用，src/ 都在 sys.path 上——batch_run 和 viz.schema 都依赖这一点。

用法
----
    python src/serve.py                  # http://127.0.0.1:5000
    python src/serve.py --port 5001
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from server.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
