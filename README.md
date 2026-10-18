# 骨骼追踪（bogu）· 后端与管线

单目视频 → 3D 人体重建 → funscript / TCode 脚本。

本仓库只包含**后端与管线**的源码。前端（React + Electron）、激活模块、
第三方模型与二进制由作者保留，不在本仓库。

## 仓库里有什么

```
src/                  Flask 后端、管线调用、转换、测试
docs/                 GVHMR Colab 说明
start.py / start.bat  开发启动（需要前端 web/，本仓库不含）
打包数据包.bat        生成可分发数据包（需要先补齐第三方依赖与前端产物）
requirements*.txt
bogu.config.example.json
```

## 没有什么

| 占位文件 | 缺的东西 | 原因 |
|---|---|---|
| `GVHMR/README_NOT_INCLUDED__GVHMR_AND_SMPLX_ARE_NONCOMMERCIAL.md` | GVHMR 源码、SMPL-X 模型 | 浙大许可：仅教育/研究/非营利；SMPL-X：非商用、需注册、禁止再分发 |
| `tools/ffmpeg/README_NOT_INCLUDED__FFMPEG_GPL_BUILD_INSTALL_SEPARATELY.md` | FFmpeg 二进制 | 常见构建是 GPLv3，随包分发需要提供源码 |
| `dist/README_NOT_INCLUDED__DATA_PACK_CONTAINS_NONCOMMERCIAL_MODELS.md` | 数据包 | 含 GVHMR / SMPL-X，不能分发 |
| `workspace/README_NOT_INCLUDED__USER_DATA_CREATED_AT_RUNTIME.md` | 素材、产物 | 用户数据，运行时生成 |

**前端 `web/` 不在本仓库。** 它是作者构建软件的位置，源码保留。

## 跑后端

### 1. Python 依赖

```bash
pip install -r requirements_win.txt
```

### 2. GVHMR + SMPL-X

- 从官方仓库安装 GVHMR 到 `GVHMR/`（非商用许可）；
- 在 SMPL-X 官网注册后下载 `SMPLX_FEMALE.npz` / `SMPLX_MALE.npz` /
  `SMPLX_NEUTRAL.npz`，放到
  `GVHMR/inputs/checkpoints/body_models/smplx/`。

这两个组件不能随商业产品分发，详见 `THIRD_PARTY_NOTICES.md`。

### 3. FFmpeg

装到 PATH（代码里是 `shutil.which("ffmpeg")`），或者把
`ffmpeg.exe` / `ffprobe.exe` 放进 `tools/ffmpeg/bin/`。

### 4. 配置

```bash
cp bogu.config.example.json bogu.config.json
# 改 root / backendPython / pipelinePython
```

### 5. 启动

只跑 API：

```bash
python src/serve.py --port 5000
# http://127.0.0.1:5000/api/health
```

`start.py` / `start.bat` 会同时启动前端开发服务器，需要 `web/`
（本仓库不含）。要完整界面，请自带前端，或从作者处获取构建产物。

## 许可证状态

- 本仓库中作者原创的代码：**版权归作者所有，尚未附加开源许可证**。
  在没有 `LICENSE` 文件之前，默认保留所有权利。
- 第三方组件的许可证与义务见 `THIRD_PARTY_NOTICES.md`。
- 缺失的组件没有被许可随本仓库分发——不要从别处拷贝后直接商用。
