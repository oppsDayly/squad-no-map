# Contributing

Contributions are welcome, but keep the current project scope in mind: this plugin is Windows x64 first, and the active OCR backends are Windows Runtime OCR plus optional Paddle Inference.

## Development Setup

Required tools:

- Windows 10/11 x64
- Visual Studio 2022 with the C++ desktop workload
- CMake 3.28 or newer
- PowerShell 7 for CI-compatible scripts

Configure and build:

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64
```

For a build that does not require Paddle Inference:

```powershell
cmake --preset windows-x64 -DENABLE_PADDLE_OCR=OFF
cmake --build --preset windows-x64
```

## Coding Rules

- Save all source and documentation files as UTF-8.
- Keep unrelated formatting churn out of functional changes.
- Do not commit `.deps/`, build output, local Paddle SDKs, model files, private screenshots, or test recordings.
- Keep OCR and rendering changes focused; OBS graphics resources need explicit cleanup paths.
- Prefer small pull requests with the build command and manual OBS test notes included.

## Assets

Do not commit private stream captures or game screenshots. If you replace embedded occluder images, document the image source and license in `THIRD_PARTY.md`.

## Pull Request Checklist

- The project configures successfully.
- The Windows x64 build succeeds.
- If OCR behavior changed, both backend modes are covered by notes or tests.
- Any new dependency or asset is documented in `THIRD_PARTY.md`.

