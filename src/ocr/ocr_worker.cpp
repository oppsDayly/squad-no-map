#include "ocr_worker.h"
#include <obs-module.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>

#if ENABLE_PADDLE_OCR
#include <paddle_inference_api.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Globalization.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Media.Ocr.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/base.h>
#endif

extern "C" void obs_log(int level, const char *format, ...);

extern "C" void pd_backfill_range(void *filter_instance, unsigned long long from,
                                  unsigned long long to, uint32_t roi_mask);

static constexpr uint32_t k_roi_bits[3] = {1u << 0, 1u << 1, 1u << 2};

namespace {

static const char *k_targets[] = {"玩", "家", "位", "置", "角", "色", "选", "择"};

static const char *backend_name(OcrBackend backend)
{
  switch (backend) {
  case OcrBackend::WindowsRuntime:
    return "WindowsRuntime";
  case OcrBackend::PaddleInference:
    return "PaddleInference";
  default:
    return "Unknown";
  }
}

static std::string trim_ascii(const std::string &s)
{
  size_t begin = 0;
  while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])))
    ++begin;
  size_t end = s.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])))
    --end;
  return s.substr(begin, end - begin);
}

static bool equals_ignore_ascii_case(const std::string &lhs, const char *rhs)
{
  if (!rhs)
    return false;
  size_t n = lhs.size();
  size_t m = std::strlen(rhs);
  if (n != m)
    return false;
  for (size_t i = 0; i < n; ++i) {
    unsigned char a = static_cast<unsigned char>(lhs[i]);
    unsigned char b = static_cast<unsigned char>(rhs[i]);
    if (std::tolower(a) != std::tolower(b))
      return false;
  }
  return true;
}

static std::string normalize_for_match(const std::string &text)
{
  std::string out;
  out.reserve(text.size());
  for (unsigned char ch : text) {
    if (!std::isspace(ch))
      out.push_back(static_cast<char>(ch));
  }
  return out;
}

static bool file_exists(const std::string &path)
{
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f)
    return false;
  std::fclose(f);
  return true;
}

static void resize_bgra_nearest(const std::vector<uint8_t> &src, int src_w, int src_h,
                                int dst_w, int dst_h, std::vector<uint8_t> &dst)
{
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
    dst.clear();
    return;
  }

  dst.resize(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 4u);
  for (int y = 0; y < dst_h; ++y) {
    int sy = std::min(src_h - 1, (y * src_h) / std::max(1, dst_h));
    for (int x = 0; x < dst_w; ++x) {
      int sx = std::min(src_w - 1, (x * src_w) / std::max(1, dst_w));
      const size_t src_idx = (static_cast<size_t>(sy) * static_cast<size_t>(src_w) +
                              static_cast<size_t>(sx)) * 4u;
      const size_t dst_idx = (static_cast<size_t>(y) * static_cast<size_t>(dst_w) +
                              static_cast<size_t>(x)) * 4u;
      dst[dst_idx + 0] = src[src_idx + 0];
      dst[dst_idx + 1] = src[src_idx + 1];
      dst[dst_idx + 2] = src[src_idx + 2];
      dst[dst_idx + 3] = src[src_idx + 3];
    }
  }
}

static bool prepare_bgra_for_winrt(const OcrRoiImage &roi, int &out_w, int &out_h,
                                   std::vector<uint8_t> &bgra)
{
  if (roi.w <= 0 || roi.h <= 0 || roi.channels < 4)
    return false;

  const size_t needed = static_cast<size_t>(roi.w) * static_cast<size_t>(roi.h) * 4u;
  if (roi.data.size() < needed)
    return false;

  out_w = roi.w;
  out_h = roi.h;
  bgra.resize(needed);
  for (size_t p = 0; p < static_cast<size_t>(roi.w) * static_cast<size_t>(roi.h); ++p) {
    const size_t i = p * 4u;
    bgra[i + 0] = roi.data[i + 2];
    bgra[i + 1] = roi.data[i + 1];
    bgra[i + 2] = roi.data[i + 0];
    bgra[i + 3] = roi.data[i + 3];
  }

  if (out_w >= 300)
    return true;

  double scale = 300.0 / static_cast<double>(std::max(1, out_w));
  if (scale > 4.0)
    scale = 4.0;
  int target_w = std::max(1, static_cast<int>(std::lround(static_cast<double>(out_w) * scale)));
  int target_h = std::max(1, static_cast<int>(std::lround(static_cast<double>(out_h) * scale)));
  if (target_w == out_w && target_h == out_h)
    return true;

  std::vector<uint8_t> scaled;
  resize_bgra_nearest(bgra, out_w, out_h, target_w, target_h, scaled);
  if (scaled.empty())
    return false;
  bgra.swap(scaled);
  out_w = target_w;
  out_h = target_h;
  return true;
}

static uint64_t quick_hash_bgra(const std::vector<uint8_t> &bgra, int w, int h)
{
  if (w <= 0 || h <= 0 || bgra.empty())
    return 0;

  constexpr uint64_t fnv_offset = 1469598103934665603ULL;
  constexpr uint64_t fnv_prime = 1099511628211ULL;
  uint64_t hash = fnv_offset;

  auto mix = [&](uint64_t value) {
    hash ^= value;
    hash *= fnv_prime;
  };

  mix(static_cast<uint64_t>(w));
  mix(static_cast<uint64_t>(h));

  const int sample_cols = std::min(16, w);
  const int sample_rows = std::min(16, h);
  const int step_x = std::max(1, w / std::max(1, sample_cols));
  const int step_y = std::max(1, h / std::max(1, sample_rows));

  for (int y = 0; y < h; y += step_y) {
    const uint8_t *row = bgra.data() + static_cast<size_t>(y) * static_cast<size_t>(w) * 4u;
    for (int x = 0; x < w; x += step_x) {
      const uint8_t *px = row + static_cast<size_t>(x) * 4u;
      uint32_t packed = 0;
      std::memcpy(&packed, px, sizeof(packed));
      mix(static_cast<uint64_t>(packed));
    }
  }

  return hash;
}

#ifdef _WIN32
static void load_dlls(const char *mode, const char *const *dlls, size_t count, bool debug_log)
{
  for (size_t i = 0; i < count; ++i) {
    const char *dll = dlls[i];
    HMODULE module = GetModuleHandleA(dll);
    if (!module)
      module = LoadLibraryA(dll);
    if (!module) {
      DWORD err = GetLastError();
      obs_log(LOG_ERROR, "[pd][ocr] LoadLibrary (%s) failed: %s (err=%lu)",
              mode, dll, (unsigned long)err);
    } else if (debug_log) {
      obs_log(LOG_INFO, "[pd][ocr] Loaded %s dep: %s", mode, dll);
    }
  }
}

static void probe_optional_dlls(const char *const *dlls, size_t count, bool debug_log)
{
  if (!debug_log)
    return;

  for (size_t i = 0; i < count; ++i) {
    const char *dll = dlls[i];
    HMODULE module = GetModuleHandleA(dll);
    if (!module)
      module = LoadLibraryA(dll);
    if (module) {
      obs_log(LOG_INFO, "[pd][ocr] Optional dep loaded: %s", dll);
    } else {
      DWORD err = GetLastError();
      obs_log(LOG_WARNING, "[pd][ocr] Optional dep not loaded: %s (err=%lu)",
              dll, (unsigned long)err);
    }
  }
}
#endif

} // namespace

struct OcrWorker::Impl {
#ifdef _WIN32
  winrt::Windows::Media::Ocr::OcrEngine winrt_engine{nullptr};
#endif
};

OcrWorker::OcrWorker(void *filter_instance) : filter_instance_(filter_instance) {}

OcrWorker::~OcrWorker()
{
  stop();
}

void OcrWorker::start()
{
  std::lock_guard<std::mutex> lk(mu_);
  if (running_)
    return;
  pending_stop_ = false;
  running_ = true;
  th_ = std::thread(&OcrWorker::run, this);
}

void OcrWorker::stop()
{
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!running_)
      return;
    pending_stop_ = true;
  }
  cv_.notify_all();
  if (th_.joinable())
    th_.join();
  running_ = false;
}

void OcrWorker::update_config(const OcrWorkerConfig &cfg)
{
  std::lock_guard<std::mutex> lk(mu_);
  OcrWorkerConfig adjusted = cfg;
  adjusted.language_tag = trim_ascii(adjusted.language_tag);
  if (adjusted.language_tag.empty())
    adjusted.language_tag = "zh-Hans-CN";
  adjusted.model_dir = trim_ascii(adjusted.model_dir);
  adjusted.dict_path = trim_ascii(adjusted.dict_path);
  adjusted.conf_threshold = std::clamp(adjusted.conf_threshold, 0.0, 1.0);
  if (adjusted.cpu_threads < 1)
    adjusted.cpu_threads = 1;
  if (adjusted.use_cpu)
    adjusted.cpu_threads = 1;
  if (adjusted.gpu_mem_mb < 1)
    adjusted.gpu_mem_mb = 512;
  if (adjusted.gpu_id < 0)
    adjusted.gpu_id = 0;
  if (adjusted.back_frames < 0)
    adjusted.back_frames = 0;
  if (adjusted.hold_frames < 0)
    adjusted.hold_frames = 0;

  bool reinit = cfg_.enable != adjusted.enable || cfg_.backend != adjusted.backend;
  if (adjusted.backend == OcrBackend::WindowsRuntime) {
    reinit = reinit || cfg_.language_tag != adjusted.language_tag;
  } else {
    reinit = reinit || cfg_.use_cpu != adjusted.use_cpu ||
             cfg_.gpu_id != adjusted.gpu_id ||
             cfg_.gpu_mem_mb != adjusted.gpu_mem_mb ||
             cfg_.model_dir != adjusted.model_dir ||
             cfg_.dict_path != adjusted.dict_path ||
             cfg_.cpu_threads != adjusted.cpu_threads;
  }
  if (reinit)
    need_reinit_ = true;

  cfg_ = std::move(adjusted);
}

void OcrWorker::submit(uint64_t frame_index, const std::array<OcrRoiImage,3> &rois)
{
  std::lock_guard<std::mutex> lk(mu_);
  if (!cfg_.enable || !running_)
    return;

  Job job;
  job.idx = frame_index;
  job.back_frames = cfg_.back_frames;
  job.hold_frames = cfg_.hold_frames;
  job.rois = rois;

  constexpr size_t kMaxQueueDepth = 2;
  while (q_.size() >= kMaxQueueDepth)
    q_.pop();
  q_.push(std::move(job));
  cv_.notify_one();
}

void OcrWorker::run()
{
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return pending_stop_ || !q_.empty(); });
      if (pending_stop_)
        break;
      job = std::move(q_.front());
      q_.pop();
    }

    uint32_t hits_mask = infer_and_match(job.rois);
    if (!hits_mask)
      continue;

    int back_frames = job.back_frames >= 0 ? job.back_frames : 0;
    int hold_frames = job.hold_frames >= 0 ? job.hold_frames : 0;
    uint64_t from = job.idx;
    if (back_frames > 0) {
      uint64_t back = (uint64_t)back_frames;
      from = (job.idx > back) ? (job.idx - back) : 0ULL;
    }
    uint64_t to = job.idx + (uint64_t)hold_frames;
    if (to < job.idx)
      to = job.idx;
    pd_backfill_range(filter_instance_, from, to, hits_mask);
  }
}

bool OcrWorker::ensure_init()
{
  OcrWorkerConfig cfg_snapshot;
  bool need_reinit_snapshot = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    cfg_snapshot = cfg_;
    need_reinit_snapshot = need_reinit_;
  }

  if (!cfg_snapshot.enable)
    return false;

  if (cfg_snapshot.backend == OcrBackend::PaddleInference)
    return ensure_paddle_init(cfg_snapshot, need_reinit_snapshot);
  return ensure_winrt_init(cfg_snapshot, need_reinit_snapshot);
}

bool OcrWorker::ensure_winrt_init(const OcrWorkerConfig &cfg_snapshot, bool need_reinit_snapshot)
{
  release_paddle();

#ifdef _WIN32
  if (!impl_)
    impl_ = std::make_unique<Impl>();

  if (!need_reinit_snapshot && impl_->winrt_engine)
    return true;

  try {
    if (!apartment_initialized_) {
      try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
      } catch (const winrt::hresult_error &ex) {
        constexpr long rpc_changed_mode = static_cast<long>(0x80010106);
        if (ex.code().value != rpc_changed_mode) {
          obs_log(LOG_ERROR, "[pd][ocr] init_apartment failed: %s",
                  winrt::to_string(ex.message()).c_str());
          return false;
        }
      }
      apartment_initialized_ = true;
    }

    using winrt::Windows::Globalization::Language;
    using winrt::Windows::Media::Ocr::OcrEngine;

    const std::string lang_tag = trim_ascii(cfg_snapshot.language_tag);
    OcrEngine engine{nullptr};
    if (equals_ignore_ascii_case(lang_tag, "auto")) {
      engine = OcrEngine::TryCreateFromUserProfileLanguages();
    } else {
      engine = OcrEngine::TryCreateFromLanguage(Language(winrt::to_hstring(lang_tag)));
    }
    if (!engine)
      engine = OcrEngine::TryCreateFromLanguage(Language(L"zh-Hans-CN"));
    if (!engine)
      engine = OcrEngine::TryCreateFromLanguage(Language(L"en-US"));
    if (!engine)
      engine = OcrEngine::TryCreateFromUserProfileLanguages();

    if (!engine) {
      obs_log(LOG_ERROR,
              "[pd][ocr] Windows OCR engine unavailable. Install Chinese or English language pack.");
      return false;
    }

    impl_->winrt_engine = engine;
    reset_winrt_roi_cache();
    {
      std::lock_guard<std::mutex> lk(mu_);
      active_cfg_ = cfg_snapshot;
      need_reinit_ = false;
    }
    if (cfg_snapshot.debug_log) {
      obs_log(LOG_INFO, "[pd][ocr] Windows OCR initialized, language=%s", lang_tag.c_str());
    }
    return true;
  } catch (const winrt::hresult_error &ex) {
    obs_log(LOG_ERROR, "[pd][ocr] initialize Windows OCR failed: %s",
            winrt::to_string(ex.message()).c_str());
    return false;
  } catch (...) {
    obs_log(LOG_ERROR, "[pd][ocr] initialize Windows OCR failed: unknown exception");
    return false;
  }
#else
  if (!unsupported_logged_) {
    unsupported_logged_ = true;
    obs_log(LOG_ERROR, "[pd][ocr] Windows OCR is unavailable on current platform");
  }
  return false;
#endif
}

bool OcrWorker::ensure_paddle_init(const OcrWorkerConfig &cfg_snapshot, bool need_reinit_snapshot)
{
#if !ENABLE_PADDLE_OCR
  UNUSED_PARAMETER(cfg_snapshot);
  UNUSED_PARAMETER(need_reinit_snapshot);
  obs_log(LOG_ERROR, "[pd][ocr] Paddle OCR backend is not compiled in this build");
  return false;
#else
#ifdef _WIN32
  if (impl_)
    impl_->winrt_engine = nullptr;
#endif
  reset_winrt_roi_cache();

  if (cfg_snapshot.model_dir.empty() || cfg_snapshot.dict_path.empty()) {
    if (cfg_snapshot.debug_log) {
      obs_log(LOG_WARNING, "[pd][ocr] Paddle OCR requires model_dir and dict_path");
    }
    return false;
  }

  if (!need_reinit_snapshot && predictor_)
    return true;

  release_paddle();

  if (cfg_snapshot.debug_log) {
    obs_log(LOG_INFO, "[pd][ocr] ensure Paddle init: use_cpu=%d gpu_id=%d gpu_mem=%d model_dir=%s dict=%s",
            cfg_snapshot.use_cpu ? 1 : 0, cfg_snapshot.gpu_id, cfg_snapshot.gpu_mem_mb,
            cfg_snapshot.model_dir.c_str(), cfg_snapshot.dict_path.c_str());
#ifdef _WIN32
    char buf[32768];
    DWORD n = GetEnvironmentVariableA("PATH", buf, sizeof(buf));
    obs_log(LOG_INFO, "[pd][ocr] PATH len=%lu", (unsigned long)n);
    const char *cuda_path = std::getenv("CUDA_PATH");
    obs_log(LOG_INFO, "[pd][ocr] CUDA_PATH=%s", cuda_path ? cuda_path : "(null)");
#endif
  }

  std::string model_path = cfg_snapshot.model_dir + "/inference.pdmodel";
  std::string params_path = cfg_snapshot.model_dir + "/inference.pdiparams";
  if (!file_exists(model_path)) {
    obs_log(LOG_ERROR, "[pd][ocr] model file missing: %s", model_path.c_str());
    return false;
  }
  if (!file_exists(params_path)) {
    obs_log(LOG_ERROR, "[pd][ocr] params file missing: %s", params_path.c_str());
    return false;
  }
  if (!file_exists(cfg_snapshot.dict_path)) {
    obs_log(LOG_ERROR, "[pd][ocr] dict file missing: %s", cfg_snapshot.dict_path.c_str());
    return false;
  }

#ifdef _WIN32
  if (cfg_snapshot.use_cpu) {
    const char *cpu_dlls[] = {
        "paddle_inference.dll", "common.dll", "mkldnn.dll", "mklml.dll", "libiomp5md.dll"};
    load_dlls("CPU", cpu_dlls, OBS_COUNTOF(cpu_dlls), cfg_snapshot.debug_log);
  } else {
    const char *core_dlls[] = {"paddle_inference.dll", "common.dll"};
    load_dlls("GPU core", core_dlls, OBS_COUNTOF(core_dlls), cfg_snapshot.debug_log);
    const char *cuda_optional[] = {
        "cudart64_110.dll", "cudart64_118.dll", "cudart64_12.dll",
        "cublas64_11.dll", "cublasLt64_11.dll", "cublas64_12.dll", "cublasLt64_12.dll",
        "cudnn64_8.dll", "cudnn64_9.dll"};
    probe_optional_dlls(cuda_optional, OBS_COUNTOF(cuda_optional), cfg_snapshot.debug_log);
  }
#endif

  try {
    paddle_infer::Config config;
    config.SetModel(model_path, params_path);
    if (cfg_snapshot.use_cpu) {
      config.DisableGpu();
      config.EnableMKLDNN();
      config.SetCpuMathLibraryNumThreads(cfg_snapshot.cpu_threads > 0 ? cfg_snapshot.cpu_threads : 1);
      if (cfg_snapshot.debug_log) {
        obs_log(LOG_INFO, "[pd][ocr] Using Paddle CPU inference (threads=%d)",
                cfg_snapshot.cpu_threads);
      }
    } else {
      config.EnableUseGpu(cfg_snapshot.gpu_mem_mb, cfg_snapshot.gpu_id);
      config.EnableCUDNN();
      if (cfg_snapshot.debug_log) {
        obs_log(LOG_INFO, "[pd][ocr] Using Paddle GPU inference");
      }
    }
    config.SwitchIrOptim(true);
    config.EnableMemoryOptim();
    config.DisableGlogInfo();

    predictor_ = paddle_infer::CreatePredictor(config);
    if (!predictor_) {
      obs_log(LOG_ERROR, "[pd][ocr] CreatePredictor returned null");
      return false;
    }
    if (!load_paddle_dict(cfg_snapshot.dict_path)) {
      release_paddle();
      return false;
    }

    {
      std::lock_guard<std::mutex> lk(mu_);
      active_cfg_ = cfg_snapshot;
      need_reinit_ = false;
    }
    if (cfg_snapshot.debug_log) {
      obs_log(LOG_INFO, "[pd][ocr] Paddle predictor created successfully");
    }
    return true;
  } catch (const std::exception &ex) {
    obs_log(LOG_ERROR, "[pd][ocr] initialize Paddle failed: %s", ex.what());
    release_paddle();
    return false;
  } catch (...) {
    obs_log(LOG_ERROR, "[pd][ocr] initialize Paddle failed: unknown exception");
    release_paddle();
    return false;
  }
#endif
}

void OcrWorker::reset_winrt_roi_cache()
{
  has_roi_cache_.fill(false);
  last_roi_hash_.fill(0);
  for (auto &text : last_roi_text_)
    text.clear();
}

void OcrWorker::release_paddle()
{
#if ENABLE_PADDLE_OCR
  predictor_.reset();
#endif
  dict_.clear();
}

bool OcrWorker::load_paddle_dict(const std::string &path)
{
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    obs_log(LOG_ERROR, "[pd][ocr] failed to open dict file: %s", path.c_str());
    return false;
  }

  std::vector<std::string> keys;
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    keys.push_back(line);
  }

  dict_.clear();
  dict_.reserve(keys.size() + 1);
  dict_.push_back("");
  dict_.insert(dict_.end(), keys.begin(), keys.end());
  return dict_.size() > 1;
}

bool OcrWorker::infer_winrt_roi_text(size_t roi_idx, const OcrRoiImage &roi, std::string &text_out)
{
  text_out.clear();
  if (roi_idx >= has_roi_cache_.size())
    return false;

  if (roi.w <= 0 || roi.h <= 0 || roi.data.empty()) {
    has_roi_cache_[roi_idx] = false;
    last_roi_hash_[roi_idx] = 0;
    last_roi_text_[roi_idx].clear();
    return false;
  }

#ifdef _WIN32
  if (!impl_ || !impl_->winrt_engine)
    return false;

  int ocr_w = 0;
  int ocr_h = 0;
  std::vector<uint8_t> bgra;
  if (!prepare_bgra_for_winrt(roi, ocr_w, ocr_h, bgra))
    return false;

  uint64_t hash = quick_hash_bgra(bgra, ocr_w, ocr_h);
  if (has_roi_cache_[roi_idx] && last_roi_hash_[roi_idx] == hash) {
    text_out = last_roi_text_[roi_idx];
    return true;
  }

  try {
    using winrt::Windows::Graphics::Imaging::BitmapAlphaMode;
    using winrt::Windows::Graphics::Imaging::BitmapPixelFormat;
    using winrt::Windows::Graphics::Imaging::SoftwareBitmap;
    using winrt::Windows::Storage::Streams::DataWriter;
    using winrt::Windows::Storage::Streams::IBuffer;

    DataWriter writer;
    writer.WriteBytes(winrt::array_view<const uint8_t>(bgra.data(), bgra.data() + bgra.size()));
    IBuffer buffer = writer.DetachBuffer();
    SoftwareBitmap bitmap = SoftwareBitmap::CreateCopyFromBuffer(
        buffer, BitmapPixelFormat::Bgra8, ocr_w, ocr_h, BitmapAlphaMode::Premultiplied);

    const auto result = impl_->winrt_engine.RecognizeAsync(bitmap).get();
    text_out = trim_ascii(winrt::to_string(result.Text()));

    has_roi_cache_[roi_idx] = true;
    last_roi_hash_[roi_idx] = hash;
    last_roi_text_[roi_idx] = text_out;
    return true;
  } catch (const winrt::hresult_error &ex) {
    obs_log(LOG_WARNING, "[pd][ocr] Windows recognize failed on ROI%zu: %s",
            roi_idx + 1, winrt::to_string(ex.message()).c_str());
    has_roi_cache_[roi_idx] = false;
    last_roi_hash_[roi_idx] = 0;
    last_roi_text_[roi_idx].clear();
    return false;
  } catch (...) {
    obs_log(LOG_WARNING, "[pd][ocr] Windows recognize failed on ROI%zu: unknown exception",
            roi_idx + 1);
    has_roi_cache_[roi_idx] = false;
    last_roi_hash_[roi_idx] = 0;
    last_roi_text_[roi_idx].clear();
    return false;
  }
#else
  return false;
#endif
}

void OcrWorker::resize_rgba_to_chw32(const OcrRoiImage &in, int out_h, int &out_w,
                                     std::vector<float> &out)
{
  if (in.w <= 0 || in.h <= 0 || in.data.empty()) {
    out_w = 0;
    out.clear();
    return;
  }

  out_h = std::max(1, out_h);
  out_w = std::max(1, (int)std::round((double)out_h * (double)in.w / (double)in.h));
  out.resize((size_t)3 * (size_t)out_h * (size_t)out_w);

  const int src_w = in.w;
  const int src_h = in.h;
  const uint8_t *src = in.data.data();
  double scale_x = (double)src_w / (double)out_w;
  double scale_y = (double)src_h / (double)out_h;
  auto access = [&](int y, int x) -> const uint8_t * {
    int sx = std::min(src_w - 1, std::max(0, (int)std::floor(x * scale_x)));
    int sy = std::min(src_h - 1, std::max(0, (int)std::floor(y * scale_y)));
    return &src[(size_t)(sy * src_w + sx) * 4];
  };

  const float mean[3] = {0.5f, 0.5f, 0.5f};
  const float stdv[3] = {0.5f, 0.5f, 0.5f};
  const float inv_std[3] = {1.0f / stdv[0], 1.0f / stdv[1], 1.0f / stdv[2]};
  float *c0 = out.data();
  float *c1 = c0 + (size_t)out_h * (size_t)out_w;
  float *c2 = c1 + (size_t)out_h * (size_t)out_w;

  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const uint8_t *p = access(y, x);
      const float b = ((float)p[2] / 255.0f - mean[0]) * inv_std[0];
      const float g = ((float)p[1] / 255.0f - mean[1]) * inv_std[1];
      const float r = ((float)p[0] / 255.0f - mean[2]) * inv_std[2];
      const size_t idx = (size_t)y * (size_t)out_w + (size_t)x;
      c0[idx] = b;
      c1[idx] = g;
      c2[idx] = r;
    }
  }
}

void OcrWorker::pad_width(std::vector<float> &tensor, int c, int h, int cur_w, int pad_w)
{
  if (pad_w <= cur_w)
    return;

  const size_t src_plane = (size_t)h * (size_t)cur_w;
  const size_t dst_plane = (size_t)h * (size_t)pad_w;
  std::vector<float> dst((size_t)c * dst_plane, 0.0f);
  for (int cc = 0; cc < c; ++cc) {
    const float *src_c = tensor.data() + (size_t)cc * src_plane;
    float *dst_c = dst.data() + (size_t)cc * dst_plane;
    for (int y = 0; y < h; ++y) {
      const float *src_row = src_c + (size_t)y * (size_t)cur_w;
      float *dst_row = dst_c + (size_t)y * (size_t)pad_w;
      std::memcpy(dst_row, src_row, sizeof(float) * (size_t)cur_w);
    }
  }
  tensor.swap(dst);
}

uint32_t OcrWorker::infer_winrt_and_match(const std::array<OcrRoiImage,3> &rois,
                                          const OcrWorkerConfig &cfg_snapshot)
{
  uint32_t hits_mask = 0;
  for (size_t i = 0; i < rois.size(); ++i) {
    std::string text;
    if (!infer_winrt_roi_text(i, rois[i], text))
      continue;

    const std::string normalized = normalize_for_match(text);
    const double conf = normalized.empty() ? 0.0 : 1.0;
    if (cfg_snapshot.debug_log) {
      obs_log(LOG_INFO, "[pd][ocr] Windows ROI%zu text='%s' conf=%.3f",
              i + 1, text.c_str(), conf);
    }
    if (normalized.empty() || conf < cfg_snapshot.conf_threshold)
      continue;

    bool matched = false;
    for (const char *needle : k_targets) {
      if (normalized.find(needle) != std::string::npos) {
        matched = true;
        break;
      }
    }
    if (matched)
      hits_mask |= k_roi_bits[i];
  }
  return hits_mask;
}

uint32_t OcrWorker::infer_paddle_and_match(const std::array<OcrRoiImage,3> &rois,
                                           const OcrWorkerConfig &cfg_snapshot)
{
#if !ENABLE_PADDLE_OCR
  UNUSED_PARAMETER(rois);
  UNUSED_PARAMETER(cfg_snapshot);
  return 0;
#else
  if (!predictor_)
    return 0;

  struct Sample {
    int w = 0;
    std::vector<float> data;
  } samples[3];

  int H = 48;
  int Wmax = 0;
  int N = 0;
  for (int i = 0; i < 3; ++i) {
    if (rois[i].w <= 0 || rois[i].h <= 0 || rois[i].data.empty())
      continue;
    resize_rgba_to_chw32(rois[i], H, samples[i].w, samples[i].data);
    Wmax = std::max(Wmax, samples[i].w);
    ++N;
  }
  if (N == 0 || Wmax <= 0)
    return 0;
  if (cfg_snapshot.debug_log) {
    obs_log(LOG_INFO, "[pd][ocr] Paddle batch N=%d H=%d Wmax=%d", N, H, Wmax);
  }

  int Wpad = (Wmax + 7) & ~7;
  for (int i = 0; i < 3; ++i) {
    if (!samples[i].data.empty())
      pad_width(samples[i].data, 3, H, samples[i].w, Wpad);
  }

  std::vector<float> input;
  input.reserve((size_t)N * 3 * H * Wpad);
  std::vector<int> batch_index;
  batch_index.reserve(N);
  for (int i = 0; i < 3; ++i) {
    if (samples[i].data.empty())
      continue;
    batch_index.push_back(i);
    input.insert(input.end(), samples[i].data.begin(), samples[i].data.end());
  }

  if (cfg_snapshot.debug_log && !batch_index.empty()) {
    const int w0 = samples[batch_index[0]].w;
    const size_t plane = (size_t)H * (size_t)Wpad;
    const float *base = input.data();
    double sumB = 0.0;
    double sumG = 0.0;
    double sumR = 0.0;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < w0; ++x) {
        const size_t idx = (size_t)y * (size_t)Wpad + (size_t)x;
        sumB += base[idx];
        sumG += base[plane + idx];
        sumR += base[2 * plane + idx];
      }
    }
    const double denom = std::max(1.0, (double)H * (double)w0);
    obs_log(LOG_INFO, "[pd][ocr] Paddle sample0 mean(B,G,R)=(%.3f, %.3f, %.3f), w=%d padW=%d",
            (float)(sumB / denom), (float)(sumG / denom), (float)(sumR / denom), w0, Wpad);
  }

  try {
    auto input_names = predictor_->GetInputNames();
    auto input_handle = predictor_->GetInputHandle(input_names[0]);
    std::vector<int> shape = {N, 3, H, Wpad};
    input_handle->Reshape(shape);
    input_handle->CopyFromCpu(input.data());

    if (cfg_snapshot.debug_log)
      obs_log(LOG_INFO, "[pd][ocr] Paddle predictor->Run begin");
    predictor_->Run();
    if (cfg_snapshot.debug_log)
      obs_log(LOG_INFO, "[pd][ocr] Paddle predictor->Run done");

    auto output_names = predictor_->GetOutputNames();
    auto output_handle = predictor_->GetOutputHandle(output_names[0]);
    std::vector<int> out_shape = output_handle->shape();
    if (out_shape.size() != 3)
      return 0;

    int nN = (int)out_shape[0];
    int dim1 = (int)out_shape[1];
    int dim2 = (int)out_shape[2];
    bool layout_NTC = dim2 > dim1;
    int T = layout_NTC ? dim1 : dim2;
    int C = layout_NTC ? dim2 : dim1;
    if (cfg_snapshot.debug_log) {
      obs_log(LOG_INFO, "[pd][ocr] Paddle out shape: N=%d, %s dims=(%d,%d), T=%d C=%d",
              nN, layout_NTC ? "NTC" : "NCT", dim1, dim2, T, C);
    }

    std::vector<float> out((size_t)nN * (size_t)T * (size_t)C);
    output_handle->CopyToCpu(out.data());

    auto at = [&](int n, int t, int c) -> float & {
      if (layout_NTC)
        return out[((size_t)n * T + t) * C + c];
      return out[((size_t)n * C + c) * T + t];
    };

    uint32_t hits_mask = 0;
    const int blank_id = 0;
    for (int n = 0; n < nN; ++n) {
      std::string s_out;
      int prev = -1;
      double conf_sum = 0.0;
      int conf_cnt = 0;
      for (int t = 0; t < T; ++t) {
        int arg = 0;
        float best = at(n, t, 0);
        float second = -std::numeric_limits<float>::infinity();
        for (int c = 1; c < C; ++c) {
          float v = at(n, t, c);
          if (v > best) {
            second = best;
            best = v;
            arg = c;
          } else if (v > second) {
            second = v;
          }
        }
        if (arg != blank_id && arg != prev) {
          double margin = (double)best - (double)second;
          if (!std::isfinite(margin))
            margin = 20.0;
          margin = std::clamp(margin, -20.0, 20.0);
          double conf_char = 1.0 / (1.0 + std::exp(-margin));
          conf_sum += conf_char;
          ++conf_cnt;
          int dict_idx = arg - 1;
          if (dict_idx >= 0 && dict_idx + 1 < (int)dict_.size())
            s_out += dict_[dict_idx + 1];
        }
        prev = arg;
      }

      double conf_avg = conf_cnt ? (conf_sum / conf_cnt) : 0.0;
      if (cfg_snapshot.debug_log) {
        obs_log(LOG_INFO, "[pd][ocr] Paddle n=%d text='%s' conf=%.3f",
                n, s_out.c_str(), conf_avg);
      }

      bool matched = false;
      if (!s_out.empty() && conf_avg >= cfg_snapshot.conf_threshold) {
        for (const char *needle : k_targets) {
          if (s_out.find(needle) != std::string::npos) {
            matched = true;
            break;
          }
        }
      }
      if (matched && n < (int)batch_index.size()) {
        int roi_idx = batch_index[n];
        if (roi_idx >= 0 && roi_idx < 3) {
          hits_mask |= k_roi_bits[roi_idx];
          if (cfg_snapshot.debug_log) {
            obs_log(LOG_INFO, "[pd][ocr] Paddle ROI%d matched, mask=0x%X",
                    roi_idx + 1, hits_mask);
          }
        }
      }
    }
    return hits_mask;
  } catch (const std::exception &ex) {
    obs_log(LOG_WARNING, "[pd][ocr] Paddle infer failed: %s", ex.what());
    return 0;
  } catch (...) {
    obs_log(LOG_WARNING, "[pd][ocr] Paddle infer failed: unknown exception");
    return 0;
  }
#endif
}

uint32_t OcrWorker::infer_and_match(const std::array<OcrRoiImage,3> &rois)
{
  if (!ensure_init())
    return 0;

  OcrWorkerConfig cfg_snapshot;
  {
    std::lock_guard<std::mutex> lk(mu_);
    cfg_snapshot = cfg_;
  }

  if (cfg_snapshot.debug_log) {
    obs_log(LOG_INFO, "[pd][ocr] infer backend=%s", backend_name(cfg_snapshot.backend));
  }

  if (cfg_snapshot.backend == OcrBackend::PaddleInference)
    return infer_paddle_and_match(rois, cfg_snapshot);
  return infer_winrt_and_match(rois, cfg_snapshot);
}
