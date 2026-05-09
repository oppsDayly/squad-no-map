# Squad No Map

[中文说明](README.zh-CN.md)

Squad No Map is an OBS Studio video filter for delaying a game capture and applying automatic occlusion when OCR detects sensitive UI text. It was built for Squad streams where the map, player list, role selection, or similar tactical information should not be exposed in real time.

The plugin registers the OBS filter `predictive_delay` and appears in the UI as `预测遮挡延迟`.

## Features

- GPU frame delay implemented with an OBS texture queue.
- Retroactive occlusion: OCR can trigger masking on a configurable frame range.
- Three ROI presets for text detection, including 16:9 and 21:9 layouts.
- Occlusion modes: embedded image, mosaic, and downsampled blur.
- OCR backends:
  - Windows Runtime OCR.
  - Paddle Inference OCR, optional at build time.
- UTF-8 source and UI text support.

## Current Scope

This project currently targets Windows x64. The repository still contains some cross-platform files inherited from the OBS plugin template, but the OCR implementation, dependency setup, CI build, and packaging flow are Windows-oriented.

## Repository Layout

- `src/plugin-main.cpp` - OBS module entry point.
- `src/filters/predictive_delay_filter.cpp` - filter implementation, delay queue, ROI capture, and occlusion rendering.
- `src/ocr/ocr_worker.cpp` - OCR worker thread and backend switching.
- `src/filters/occluder_builtin.h` - embedded default occluder images as byte arrays.
- `tools/img.py` - helper script for regenerating embedded occluder image data.
- `cmake/`, `build-aux/`, `.github/` - OBS plugin template build and packaging support.

## Source Builds and Artifacts

The repository intentionally excludes:

- `.deps/` downloaded OBS/Paddle SDK files.
- `build*/`, `release/`, and other generated build output.
- Local screenshots, test media, model files, and private image exports.

Run CMake once to download the OBS prebuilt dependencies used by the template. Paddle Inference is local-only and must be supplied separately if you enable that backend.

## Requirements

- Windows 10/11 x64.
- OBS Studio 31.x runtime for using the plugin.
- Visual Studio 2022 with the C++ desktop workload.
- CMake 3.28 or newer.
- PowerShell 7 for template CI scripts.

For the Windows Runtime OCR backend:

- The corresponding Windows OCR language pack, for example Chinese Simplified for `zh-Hans-CN`.

For the Paddle backend:

- Paddle Inference SDK installed at `.deps/paddle31`.
- A PaddleOCR recognition model directory containing `inference.pdmodel` and `inference.pdiparams`.
- A PaddleOCR dictionary file such as `ppocr_keys_v1.txt`.
- CUDA/cuDNN installed and visible on `PATH` when using Paddle GPU mode.

The Paddle SDK is intentionally not committed. The project builds without it by disabling Paddle OCR.

## Configure and Build

Default Windows build. If `.deps/paddle31` is present, Paddle OCR is enabled automatically; otherwise the build falls back to Windows Runtime OCR only.

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64
```

Build without Paddle OCR:

```powershell
cmake --preset windows-x64 -DENABLE_PADDLE_OCR=OFF
cmake --build --preset windows-x64
```

Build with Paddle OCR:

```powershell
cmake --preset windows-x64 -DENABLE_PADDLE_OCR=ON
cmake --build --preset windows-x64
```

When Paddle OCR is enabled, CMake expects `.deps/paddle31/paddle/include/paddle_inference_api.h` and the related Paddle libraries to exist.

The built plugin DLL is written to:

```text
build_x64/RelWithDebInfo/squad-no-map.dll
```

Runtime files are copied to:

```text
build_x64/rundir/RelWithDebInfo/
```

## OBS Usage

1. Install or copy the built plugin into the OBS plugin directory.
2. Add the `预测遮挡延迟` filter to the video source that contains the game capture.
3. Set `延迟`, `回溯帧数`, and `保持帧数`.
4. Choose a `识别引擎`:
   - `Windows Runtime OCR` for system OCR.
   - `Paddle Inference CPU` or `Paddle Inference GPU` for PaddleOCR.
5. If using Paddle, set the model directory and dictionary path.
6. Adjust the resolution preset and occlusion mode.

## Regenerating Embedded Occluder Images

Use `tools/img.py` to replace the three embedded occluder images:

```powershell
python tools/img.py roi1.png roi2.png roi3.png
```

This rewrites `src/filters/occluder_builtin.h` using UTF-8.

Do not commit private screenshots or local test assets from `tools/`; `.gitignore` only allows `tools/img.py`.

## Licensing

The project is licensed under GPL-2.0-or-later as stated in source headers. The repository includes the GPL-2.0 license text in `LICENSE`.

Third-party dependencies have their own licenses. See `THIRD_PARTY.md`.

Before publishing a release, review `buildspec.json` and `src/filters/occluder_builtin.h` so the public author metadata and embedded image assets are intentional.
