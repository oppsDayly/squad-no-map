#include "ocr_worker.h"
#include <obs-module.h>
#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>
#include <memory>
#include <limits>
#include <algorithm>
#include <cmath>

#include <paddle_inference_api.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

extern "C" void obs_log(int level, const char *format, ...);

extern "C" void pd_backfill_now(void *filter_instance, int back_frames, int hold_frames);
extern "C" void pd_backfill_range(void *filter_instance, unsigned long long from, unsigned long long to);

OcrWorker::OcrWorker(void *filter_instance) : filter_instance_(filter_instance) {}
OcrWorker::~OcrWorker() { stop(); }

void OcrWorker::start() {
  std::lock_guard<std::mutex> lk(mu_);
  if (running_) return;
  pending_stop_ = false;
  running_ = true;
  th_ = std::thread(&OcrWorker::run, this);
}

void OcrWorker::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!running_) return;
    pending_stop_ = true;
  }
  cv_.notify_all();
  if (th_.joinable()) th_.join();
  running_ = false;
}

void OcrWorker::update_config(const OcrWorkerConfig &cfg) {
  std::lock_guard<std::mutex> lk(mu_);
  if (cfg_.enable != cfg.enable || cfg_.gpu_id != cfg.gpu_id || cfg_.gpu_mem_mb != cfg.gpu_mem_mb ||
      cfg_.model_dir != cfg.model_dir || cfg_.dict_path != cfg.dict_path ||
      cfg_.conf_threshold != cfg.conf_threshold ||
      cfg_.use_cpu != cfg.use_cpu || cfg_.cpu_threads != cfg.cpu_threads) {
    need_reinit_ = true;
  }
  cfg_ = cfg;
}

void OcrWorker::submit(uint64_t frame_index, const std::array<OcrRoiImage,3> &rois) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!cfg_.enable) return;
  if (!running_) return;
  Job job;
  job.idx = frame_index;
  job.back_frames = cfg_.back_frames;
  job.hold_frames = cfg_.hold_frames;
  job.cpu_threads = cfg_.cpu_threads;
  job.rois = rois;
  q_.push(std::move(job));
  cv_.notify_one();
}

void OcrWorker::run() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&]{ return pending_stop_ || !q_.empty(); });
      if (pending_stop_) break;
      job = std::move(q_.front());
      q_.pop();
    }

    bool hit = infer_and_match(job.rois);
    if (hit) {
      int back_frames = job.back_frames >= 0 ? job.back_frames : 0;
      int hold_frames = job.hold_frames >= 0 ? job.hold_frames : 0;
      uint64_t from = job.idx;
      if (back_frames > 0) {
        uint64_t back = (uint64_t)back_frames;
        from = (job.idx > back) ? (job.idx - back) : 0ULL;
      }
      uint64_t to = job.idx + (uint64_t)hold_frames;
      if (to < job.idx) to = job.idx; // handle overflow
      pd_backfill_range(filter_instance_, from, to);
    }
  }
}

bool OcrWorker::ensure_init() {
  if (!cfg_.enable) return false;
  if (cfg_.model_dir.empty() || cfg_.dict_path.empty()) return false;
  if (!need_reinit_ && predictor_) return true;

  if (cfg_.debug_log) {
    obs_log(LOG_INFO, "[pd][ocr] ensure_init: gpu_id=%d, gpu_mem=%d, model_dir=%s, dict=%s", cfg_.gpu_id, cfg_.gpu_mem_mb, cfg_.model_dir.c_str(), cfg_.dict_path.c_str());
#ifdef _WIN32
    char buf[32768]; DWORD n = GetEnvironmentVariableA("PATH", buf, sizeof(buf));
    obs_log(LOG_INFO, "[pd][ocr] PATH len=%lu", (unsigned long)n);
    const char *cuda_path = getenv("CUDA_PATH");
    obs_log(LOG_INFO, "[pd][ocr] CUDA_PATH=%s", cuda_path ? cuda_path : "(null)");
#endif
  }

  // Preflight: check files exist
  std::string model_path = cfg_.model_dir + "/inference.pdmodel";
  std::string params_path = cfg_.model_dir + "/inference.pdiparams";
  auto exists = [](const std::string &p)->bool { FILE *f = fopen(p.c_str(), "rb"); if (!f) return false; fclose(f); return true; };
  if (!exists(model_path)) { obs_log(LOG_ERROR, "[pd][ocr] model file missing: %s", model_path.c_str()); return false; }
  if (!exists(params_path)) { obs_log(LOG_ERROR, "[pd][ocr] params file missing: %s", params_path.c_str()); return false; }
  if (!exists(cfg_.dict_path)) { obs_log(LOG_ERROR, "[pd][ocr] dict file missing: %s", cfg_.dict_path.c_str()); return false; }

#ifdef _WIN32
  // 预加载必要的 Paddle 运行时 DLL，尽早暴露缺失依赖（根据 CPU/GPU 模式区分）
  if (cfg_.use_cpu) {
    const char *dlls[] = {"paddle_inference.dll","common.dll","mkldnn.dll","mklml.dll","libiomp5md.dll"};
    for (const char *d : dlls) {
      HMODULE m = GetModuleHandleA(d);
      if (!m) {
        m = LoadLibraryA(d);
        if (!m) {
          DWORD e = GetLastError();
          obs_log(LOG_ERROR, "[pd][ocr] LoadLibrary (CPU) failed: %s (err=%lu)", d, (unsigned long)e);
        } else if (cfg_.debug_log) {
          obs_log(LOG_INFO, "[pd][ocr] Loaded CPU dep: %s", d);
        }
      } else if (cfg_.debug_log) {
        obs_log(LOG_INFO, "[pd][ocr] CPU dep already loaded: %s", d);
      }
    }
  } else {
      // GPU 基础依赖（其余 CUDA/cuDNN 仅做可选探测，Paddle 3.1 不再附带 TensorRT）
    const char *core_dlls[] = {"paddle_inference.dll","common.dll"};
    for (const char *d : core_dlls) {
      HMODULE m = GetModuleHandleA(d);
      if (!m) {
        m = LoadLibraryA(d);
        if (!m) {
          DWORD e = GetLastError();
          obs_log(LOG_ERROR, "[pd][ocr] LoadLibrary (GPU core) failed: %s (err=%lu)", d, (unsigned long)e);
        } else if (cfg_.debug_log) {
          obs_log(LOG_INFO, "[pd][ocr] Loaded GPU core dep: %s", d);
        }
      } else if (cfg_.debug_log) {
        obs_log(LOG_INFO, "[pd][ocr] GPU core dep already loaded: %s", d);
      }
    }
      // 可选探测：CUDA/cuDNN（Paddle 3.1 不再捆绑 TensorRT）
    const char *cuda_optional[] = {
        "cudart64_110.dll", "cudart64_118.dll", "cudart64_12.dll",
        "cublas64_11.dll", "cublasLt64_11.dll", "cublas64_12.dll", "cublasLt64_12.dll",
        "cudnn64_8.dll", "cudnn64_9.dll"
    };
    for (const char *d : cuda_optional) {
      HMODULE m = GetModuleHandleA(d);
      if (!m) {
        m = LoadLibraryA(d);
        DWORD e = GetLastError();
        if (!m && cfg_.debug_log) obs_log(LOG_WARNING, "[pd][ocr] Optional dep not loaded: %s (err=%lu)", d, (unsigned long)e);
        else if (m && cfg_.debug_log) obs_log(LOG_INFO, "[pd][ocr] Loaded optional dep: %s", d);
      } else if (cfg_.debug_log) {
        obs_log(LOG_INFO, "[pd][ocr] Optional dep already loaded: %s", d);
      }
    }
  }
#endif

  paddle_infer::Config config;
  config.SetModel(model_path, params_path);
  obs_log(LOG_INFO, "[pd][ocr] Config prepared: use_cpu=%d gpu_id=%d gpu_mem=%d", cfg_.use_cpu ? 1 : 0, cfg_.gpu_id, cfg_.gpu_mem_mb);
  if (cfg_.use_cpu) {
    config.DisableGpu();
    config.EnableMKLDNN();
    int threads = cfg_.cpu_threads > 0 ? cfg_.cpu_threads : 1;
    config.SetCpuMathLibraryNumThreads(threads);
    if (cfg_.debug_log) obs_log(LOG_INFO, "[pd][ocr] Using CPU inference (threads=%d)", threads);
  } else {
    config.EnableUseGpu(cfg_.gpu_mem_mb, cfg_.gpu_id);
    obs_log(LOG_INFO, "[pd][ocr] EnableUseGpu done");
    config.EnableCUDNN();
    if (cfg_.debug_log) obs_log(LOG_INFO, "[pd][ocr] Using GPU inference");
  }
  config.SwitchIrOptim(true);
  config.EnableMemoryOptim();
  config.DisableGlogInfo();
  obs_log(LOG_INFO, "[pd][ocr] CreatePredictor begin");
  predictor_ = paddle_infer::CreatePredictor(config);
  obs_log(LOG_INFO, "[pd][ocr] CreatePredictor end: %s", predictor_ ? "ok" : "null");
  if (!predictor_) {
    obs_log(LOG_ERROR, "[pd][ocr] CreatePredictor returned null");
    return false;
  }
  if (!load_dict(cfg_.dict_path)) return false;
  active_cfg_ = cfg_;
  need_reinit_ = false;
  if (cfg_.debug_log) obs_log(LOG_INFO, "[pd][ocr] predictor created successfully");
  return true;
}

bool OcrWorker::load_dict(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) return false;
  std::vector<std::string> keys;
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    keys.push_back(line);
  }
  // Insert blank at index 0 for CTC
  dict_.clear();
  dict_.reserve(keys.size() + 1);
  dict_.push_back("");
  dict_.insert(dict_.end(), keys.begin(), keys.end());
  return !dict_.empty();
}

void OcrWorker::resize_rgba_to_chw32(const OcrRoiImage &in, int out_h, int &out_w, std::vector<float> &out) {
  if (in.w <= 0 || in.h <= 0) { out_w = 0; out.clear(); return; }
  out_h = std::max(1, out_h);
  out_w = std::max(1, (int)std::round((double)out_h * (double)in.w / (double)in.h));
  out.resize((size_t)3 * (size_t)out_h * (size_t)out_w);
  const int src_w = in.w;
  const int src_h = in.h;
  const uint8_t *src = in.data.data();
  double scale_x = (double)src_w / (double)out_w;
  double scale_y = (double)src_h / (double)out_h;
  auto access = [&](int y, int x)->const uint8_t*{
    int sx = std::min(src_w - 1, std::max(0, (int)std::floor(x * scale_x)));
    int sy = std::min(src_h - 1, std::max(0, (int)std::floor(y * scale_y)));
    return &src[(size_t)(sy * src_w + sx) * 4];
  };
  // Windows 下 OBS stagesurface 多数是 BGRA 内存序，PaddleOCR 识别默认按 BGR 通道；
  // 归一化使用 ((x/255.0) - mean) / std，其中 mean=[0.5,0.5,0.5]，std=[0.5,0.5,0.5]
  const float mean[3] = {0.5f, 0.5f, 0.5f};
  const float stdv[3] = {0.5f, 0.5f, 0.5f};
  const float inv_std[3] = {1.0f/stdv[0], 1.0f/stdv[1], 1.0f/stdv[2]};
  float *c0 = out.data();
  float *c1 = c0 + (size_t)out_h * (size_t)out_w;
  float *c2 = c1 + (size_t)out_h * (size_t)out_w;
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const uint8_t *p = access(y, x);
      // 映射 RGBA -> BGR（忽略 A），OBS 阶段数据为 RGBA
      const float b = ((float)p[2] / 255.0f - mean[0]) * inv_std[0];
      const float g = ((float)p[1] / 255.0f - mean[1]) * inv_std[1];
      const float r = ((float)p[0] / 255.0f - mean[2]) * inv_std[2];
      const size_t idx = (size_t)y * (size_t)out_w + (size_t)x;
      // CHW 顺序，通道顺序按 B,G,R 存放
      c0[idx] = b; c1[idx] = g; c2[idx] = r;
    }
  }
}

void OcrWorker::pad_width(std::vector<float> &tensor, int c, int h, int cur_w, int pad_w) {
  if (pad_w <= cur_w) return;
  // 将 CHW 排列的张量逐行右侧补零，保证每个通道的每一行从 cur_w 扩展到 pad_w。
  const size_t src_plane = (size_t)h * (size_t)cur_w;
  const size_t dst_plane = (size_t)h * (size_t)pad_w;
  std::vector<float> dst((size_t)c * dst_plane, 0.0f);
  for (int cc = 0; cc < c; ++cc) {
    const float *src_c = tensor.data() + (size_t)cc * src_plane;
    float *dst_c = dst.data() + (size_t)cc * dst_plane;
    for (int y = 0; y < h; ++y) {
      const float *src_row = src_c + (size_t)y * (size_t)cur_w;
      float *dst_row = dst_c + (size_t)y * (size_t)pad_w;
      // 拷贝每行有效宽度
      std::memcpy(dst_row, src_row, sizeof(float) * (size_t)cur_w);
      // 其余部分默认 0 已经作为 padding
    }
  }
  tensor.swap(dst);
}

bool OcrWorker::infer_and_match(const std::array<OcrRoiImage,3> &rois) {
  if (!ensure_init()) return false;
  // Build batch
  struct Sample { int w=0; std::vector<float> data; } s[3];
  int H = 48; int Wmax = 0; int N = 0;
  for (int i = 0; i < 3; ++i) {
    if (rois[i].w <= 0 || rois[i].h <= 0 || rois[i].data.empty()) continue;
    resize_rgba_to_chw32(rois[i], H, s[i].w, s[i].data);
    Wmax = std::max(Wmax, s[i].w);
    ++N;
  }
  if (N == 0 || Wmax <= 0) return false;
  if (cfg_.debug_log) {
    obs_log(LOG_INFO, "[pd][ocr] batch N=%d H=%d Wmax=%d", N, H, Wmax);
  }
  // Align width to 8 for better kernel compatibility
  int Wpad = (Wmax + 7) & ~7;
  // pad each to Wmax
  for (int i = 0; i < 3; ++i) if (!s[i].data.empty()) pad_width(s[i].data, 3, H, s[i].w, Wpad);

  // pack batch: [N,3,H,Wmax]
  std::vector<float> input;
  input.reserve((size_t)N * 3 * H * Wpad);
  std::vector<int> batch_index; batch_index.reserve(N);
  for (int i = 0; i < 3; ++i) {
    if (s[i].data.empty()) continue;
    batch_index.push_back(i);
    input.insert(input.end(), s[i].data.begin(), s[i].data.end());
  }

  // 调试：打印第一个样本的通道均值，帮助确认通道顺序与归一化是否正确
  if (cfg_.debug_log && !batch_index.empty()) {
    const int w0 = s[batch_index[0]].w;
    const size_t plane = (size_t)H * (size_t)Wpad;
    const float *base = input.data();
    double sumB = 0.0, sumG = 0.0, sumR = 0.0;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < w0; ++x) {
        const size_t idx = (size_t)y * (size_t)Wpad + (size_t)x;
        sumB += base[idx];
        sumG += base[plane + idx];
        sumR += base[2 * plane + idx];
      }
    }
    const double denom = std::max(1.0, (double)H * (double)w0);
    obs_log(LOG_INFO, "[pd][ocr] sample0 mean(B,G,R)=(%.3f, %.3f, %.3f), w=%d padW=%d",
            (float)(sumB / denom), (float)(sumG / denom), (float)(sumR / denom), w0, Wpad);
  }

  auto input_names = predictor_->GetInputNames();
  auto input_handle = predictor_->GetInputHandle(input_names[0]);
  std::vector<int> shape = { N, 3, H, Wpad };
  input_handle->Reshape(shape);
  input_handle->CopyFromCpu(input.data());
  if (cfg_.debug_log) obs_log(LOG_INFO, "[pd][ocr] calling predictor->Run()...");
  predictor_->Run();
  if (cfg_.debug_log) obs_log(LOG_INFO, "[pd][ocr] predictor->Run() done");

  auto output_names = predictor_->GetOutputNames();
  auto output_handle = predictor_->GetOutputHandle(output_names[0]);
  std::vector<int> out_shape = output_handle->shape();
  // Expect [N, T, C] or [N, C, T]
  if (out_shape.size() != 3) return false;
  int nN = (int)out_shape[0];
  int dim1 = (int)out_shape[1];
  int dim2 = (int)out_shape[2];
  bool layout_NTC = dim2 > dim1; // heuristic: classes usually larger than time
  int T = layout_NTC ? dim1 : dim2;
  int C = layout_NTC ? dim2 : dim1;
  if (cfg_.debug_log) {
    obs_log(LOG_INFO, "[pd][ocr] out shape: N=%d, %s dims=(%d,%d), T=%d C=%d", nN, layout_NTC?"NTC":"NCT", dim1, dim2, T, C);
  }
  std::vector<float> out;
  out.resize((size_t)nN * (size_t)T * (size_t)C);
  output_handle->CopyToCpu(out.data());

  // CTC greedy decode
  auto at = [&](int n,int t,int c)->float&{
    if (layout_NTC) return out[((size_t)n*T + t)*C + c];
    else return out[((size_t)n*C + c)*T + t];
  };

  static const char *kTargets[] = {"玩", "家", "位", "置", "角", "色", "选", "择"};
  const int blank_id = 0; // PaddleOCR CTC blank = 0 (dict shifted by +1)
  for (int n = 0; n < nN; ++n) {
    std::string s_out;
    int prev = -1;
    double conf_sum = 0.0; int conf_cnt = 0;
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
        if (!std::isfinite(margin)) margin = 20.0;
        if (margin > 20.0) margin = 20.0;
        if (margin < -20.0) margin = -20.0;
        double conf_char = 1.0 / (1.0 + std::exp(-margin));
        conf_sum += conf_char; conf_cnt++;
        int dict_idx = arg - 1; // shift by 1 (blank=0)
        if (dict_idx >= 0 && dict_idx + 1 < (int)dict_.size()) s_out += dict_[dict_idx+1];
      }
      prev = arg;
    }
    double conf_avg = conf_cnt ? (conf_sum / conf_cnt) : 0.0;
    if (cfg_.debug_log) {
      obs_log(LOG_INFO, "[pd][ocr] n=%d text='%s' conf=%.3f", n, s_out.c_str(), conf_avg);
    }
    if (!s_out.empty() && conf_avg >= 0.6) {
      for (const char *needle : kTargets) {
        if (s_out.find(needle) != std::string::npos) {
          return true;
        }
      }
    }
  }
  return false;
}

