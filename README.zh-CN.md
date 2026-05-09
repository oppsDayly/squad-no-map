# Squad No Map 中文说明

[English README](README.md)

Squad No Map 是一个 OBS Studio 视频滤镜插件，用于对游戏画面做延迟输出，并在 OCR 识别到敏感 UI 文本时自动回溯遮挡。它主要面向《Squad》直播场景，避免地图、玩家列表、角色选择等战术信息被实时暴露。

插件在 OBS 中注册的滤镜 ID 为 `predictive_delay`，界面名称为 `预测遮挡延迟`。

## 功能特性

- 基于 OBS 纹理队列实现 GPU 画面延迟。
- 支持 OCR 触发后的回溯遮挡，可配置回溯帧数和保持帧数。
- 内置 3 个识别区域，支持 16:9 和 21:9 分辨率预设。
- 支持图片遮挡、马赛克遮挡和降采样模糊遮挡。
- 支持两类 OCR 后端：
  - Windows Runtime OCR。
  - Paddle Inference OCR，可选编译，支持 CPU/GPU 模式。
- 源码和界面字符串按 UTF-8 处理。

## 当前范围

当前项目优先支持 Windows x64。仓库中仍保留了一些 OBS 插件模板的跨平台构建文件，但 OCR 实现、依赖准备、CI 构建和打包流程都以 Windows 为主。

## 目录结构

- `src/plugin-main.cpp`：OBS 模块入口。
- `src/filters/predictive_delay_filter.cpp`：滤镜主体、延迟队列、ROI 截图和遮挡渲染。
- `src/ocr/ocr_worker.cpp`：OCR 后台线程和识别后端切换。
- `src/filters/occluder_builtin.h`：内置遮挡图片字节数组。
- `tools/img.py`：重新生成内置遮挡图片数据的辅助脚本。
- `cmake/`、`build-aux/`、`.github/`：基于 OBS 插件模板的构建与打包辅助文件。

## 不会提交到仓库的内容

仓库会刻意排除以下文件：

- `.deps/` 中下载或本地安装的 OBS/Paddle SDK。
- `build*/`、`release/` 等构建输出目录。
- 本地截图、测试视频、模型文件和私有图片素材。

Paddle Inference SDK 不会随仓库提交。如果不需要 Paddle 后端，可以关闭 Paddle OCR 后直接构建 Windows Runtime OCR 版本。

## 环境要求

- Windows 10/11 x64。
- OBS Studio 31.x 运行环境。
- Visual Studio 2022，并安装 C++ 桌面开发工作负载。
- CMake 3.28 或更新版本。
- PowerShell 7，用于兼容模板 CI 脚本。

Windows Runtime OCR 后端还需要安装对应的 Windows OCR 语言包，例如 `zh-Hans-CN` 对应的简体中文语言包。

Paddle 后端额外需要：

- Paddle Inference SDK，默认放在 `.deps/paddle31`。
- PaddleOCR 识别模型目录，包含 `inference.pdmodel` 和 `inference.pdiparams`。
- PaddleOCR 字典文件，例如 `ppocr_keys_v1.txt`。
- 使用 GPU 模式时，需要 CUDA/cuDNN 可用，并能在 `PATH` 中找到相关运行库。

## 构建方式

默认 Windows 构建。如果 `.deps/paddle31` 存在，会自动启用 Paddle OCR；否则只构建 Windows Runtime OCR。

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64
```

明确关闭 Paddle OCR：

```powershell
cmake --preset windows-x64 -DENABLE_PADDLE_OCR=OFF
cmake --build --preset windows-x64
```

明确启用 Paddle OCR：

```powershell
cmake --preset windows-x64 -DENABLE_PADDLE_OCR=ON
cmake --build --preset windows-x64
```

启用 Paddle OCR 时，CMake 会检查 `.deps/paddle31/paddle/include/paddle_inference_api.h` 以及相关 Paddle 库是否存在。

构建后的插件 DLL 默认位于：

```text
build_x64/RelWithDebInfo/squad-no-map.dll
```

运行测试用文件会复制到：

```text
build_x64/rundir/RelWithDebInfo/
```

## OBS 使用方式

1. 将构建出的插件安装或复制到 OBS 插件目录。
2. 在包含游戏画面的源上添加 `预测遮挡延迟` 滤镜。
3. 设置 `延迟`、`回溯帧数` 和 `保持帧数`。
4. 在 `识别引擎` 中选择：
   - `Windows Runtime OCR`：使用系统 OCR。
   - `Paddle Inference CPU`：使用 Paddle CPU 推理。
   - `Paddle Inference GPU`：使用 Paddle GPU 推理。
5. 如果选择 Paddle 后端，需要设置模型目录和字典文件路径。
6. 根据分辨率选择预设，并按需要调整遮挡模式。

## 重新生成内置遮挡图片

使用 `tools/img.py` 替换 3 张内置遮挡图：

```powershell
python tools/img.py roi1.png roi2.png roi3.png
```

脚本会用 UTF-8 重写 `src/filters/occluder_builtin.h`。

不要提交私有截图或本地测试素材；`.gitignore` 只允许提交 `tools/img.py`。

## 开源发布注意事项

- 发布前确认 `buildspec.json` 中的作者、网站和邮箱信息是否符合你的公开身份。
- 发布前确认 `src/filters/occluder_builtin.h` 内嵌图片拥有可公开发布的授权。
- 如果二进制包包含 Paddle、CUDA、cuDNN 或其他第三方运行库，需要随包保留对应许可证说明。
- 贡献代码时请确保所有源码和文档均为 UTF-8 编码。

## 许可证

项目源代码按 GPL-2.0-or-later 授权，仓库中包含 GPL-2.0 正文，见 `LICENSE`。

第三方依赖有各自的许可证说明，见 `THIRD_PARTY.md`。

