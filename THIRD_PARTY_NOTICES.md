# 第三方组件与许可证

本仓库**不分发**下表中标注为"未包含"的第三方内容。它们需要使用者自行
获取并遵守各自的许可证；其中若干**禁止商业使用或再分发**。

> 这不是法律意见。商用前请向对应权利方确认，或咨询律师。

## 一、决定能否分发 / 商用的组件

| 组件 | 许可证 | 关键限制 | 本仓库 |
|---|---|---|---|
| **GVHMR**（浙大 3D Vision Group） | 自定义：仅教育、研究、非营利 | 衍生作品必须开源；**禁止商业使用**；商用联系 `xwzhou@zju.edu.cn` | 未包含 |
| **SMPL-X 模型**（Max Planck） | 自定义：非商用、需注册 | **禁止再分发**；商用需单独授权（Meshcapade） | 未包含 |
| **osr-emu 核心**（ayvajs） | MIT | 保留版权与许可声明 | 未包含（需自行构建） |
| **OFS-SE 模型**（shadowsashimi/ofs-se） | **GPL-3.0-only** | 与 osr-emu 打进同一个包后，**整个 bundle 是 GPL-3.0** | 未包含 |
| **FFmpeg**（gyan.dev essentials） | **GPL-3.0**（`--enable-gpl --enable-version3`） | 分发二进制须提供对应源码/书面要约 | 未包含 |
| **Dujiao-Next** | GPL-3.0 | 分发二进制须提供对应源码；修改须开源 | 不在本仓库 |

### 实际含义

- **GVHMR + SMPL-X 是当前商业化的硬约束**：只要你的产品随包携带它们，
  或者把它们作为卖点交付，就不符合"非商用"条款——开不开源都不改变这一点。
  可行路径只有：取得商业授权、由用户自行安装（且用途仍受其许可约束），
  或替换成有商业许可的方案。
- **OFS-SE 是前端开闭源的分水岭**：去掉它重新打包，`osr-emu.js` 是 MIT，
  前端可以闭源；保留它，整包就是 GPL-3.0。
- **FFmpeg** 用 LGPL 构建可以避免 GPL 的传染性；用 GPL 构建则要按 GPLv3
  提供源码。最省事的做法是让用户自己安装。

## 二、宽松许可证依赖（保留声明即可）

### JavaScript / 前端

| 包 | 许可证 |
|---|---|
| electron | MIT |
| react / react-dom | MIT |
| antd / @ant-design/icons | MIT |
| three | MIT |
| zustand | MIT |
| express | MIT |
| ws | MIT |
| vite（开发） | MIT |
| typescript（开发） | Apache-2.0 |
| electron-builder（开发） | MIT |

### Python

| 包 | 许可证 |
|---|---|
| Flask | BSD-3-Clause |
| flask-cors | MIT |
| pyserial | BSD-3-Clause |
| numpy | BSD-3-Clause |
| torch | BSD-3-Clause |
| MMPose / RTMPose | Apache-2.0 |

MIT / BSD / Apache-2.0 都要求保留版权与许可声明。发布二进制或源码时，
请从各依赖的发布包中取出完整的 `LICENSE` 文本一并附上；上表只是索引。

## 三、关于前端（不在本仓库）

前端源码不在本仓库，但它使用的 `osr-emu` 有明确的许可证边界：

- osr-emu 核心（OSR2 / SR6 / SSR1）是 **MIT**；
- OFS-SE 模型是 **GPL-3.0-only**；
- 两者打进同一个 ESM bundle 后，产物整体按 **GPL-3.0** 处理。

因此：闭源分发前端时必须去掉 OFS-SE；保留 OFS-SE 则整个 bundle 与
引入它的前端都要按 GPL-3.0 处理。

## 四、本仓库原创代码

作者原创部分目前**没有附加开源许可证**（没有 `LICENSE` 文件 = 保留所有权利）。
在明确选择许可证之前，请勿假定可以复制、修改或再分发。
