# Third-Party Notices

This repository contains source code for the Squad No Map OBS plugin. It does not vendor the large SDKs, model files, or runtime packages listed below.

## OBS Studio and libobs

The plugin is built against OBS Studio/libobs headers and libraries. The build scripts download OBS prebuilt dependencies into `.deps/`.

- Project: https://github.com/obsproject/obs-studio
- License: GPL-2.0-or-later

## OBS Plugin Template Build Files

The `cmake/`, `build-aux/`, and `.github/` build support files are based on the OBS plugin template. They have been adjusted for this Windows x64 plugin.

- Project: https://github.com/obsproject/obs-plugintemplate
- License: GPL-2.0-or-later

## Windows Runtime OCR

The Windows Runtime OCR backend uses Windows APIs available on Windows 10/11. No Microsoft SDK binaries are committed to this repository.

- Component: Windows.Media.Ocr and related Windows Runtime APIs
- Provider: Microsoft Windows SDK / Windows Runtime

Users may need to install the matching Windows OCR language pack for the configured language tag.

## Paddle Inference and PaddleOCR

The Paddle backend is optional at build time. The expected local SDK location is `.deps/paddle31`, but that directory is ignored and must not be committed.

- Paddle Inference: https://github.com/PaddlePaddle/Paddle
- PaddleOCR models and dictionaries: https://github.com/PaddlePaddle/PaddleOCR

Paddle Inference packages may include additional third-party runtime libraries such as protobuf, glog, gflags, oneDNN, MKL, OpenMP, xxHash, utf8proc, CUDA, and cuDNN. Keep their license notices with any binary distribution that includes those files.

## Embedded Occluder Images

`src/filters/occluder_builtin.h` stores built-in occluder image bytes. The source images used to generate those bytes must be publishable under a license compatible with this project before release.

Use `tools/img.py` to regenerate the header from replacement images.

