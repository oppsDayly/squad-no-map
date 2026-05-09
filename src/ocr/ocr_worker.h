#pragma once

#ifndef ENABLE_PADDLE_OCR
#define ENABLE_PADDLE_OCR 0
#endif

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

struct OcrRoiImage {
  uint64_t frame_index = 0;
  int w = 0;
  int h = 0;
  int channels = 4; // RGBA
  std::vector<uint8_t> data; // tightly packed RGBA
};

enum class OcrBackend : int {
  WindowsRuntime = 0,
  PaddleInference = 1,
};

struct OcrWorkerConfig {
  bool enable = false;
  bool debug_log = false;
  OcrBackend backend = OcrBackend::WindowsRuntime;
  std::string language_tag = "zh-Hans-CN";
  bool use_cpu = false;
  int gpu_id = 0;
  int gpu_mem_mb = 512;
  std::string model_dir;   // ppocr rec model dir
  std::string dict_path;   // character dict (ppocr_keys_v1.txt)
  double conf_threshold = 0.7;
  int back_frames = 90;
  int hold_frames = 55;
  int cpu_threads = 1;
};

#if ENABLE_PADDLE_OCR
namespace paddle_infer { class Predictor; }
#endif

class OcrWorker {
public:
  explicit OcrWorker(void *filter_instance);
  ~OcrWorker();

  void start();
  void stop();
  void update_config(const OcrWorkerConfig &cfg);
  void submit(uint64_t frame_index, const std::array<OcrRoiImage,3> &rois);

private:
  void run();
  uint32_t infer_and_match(const std::array<OcrRoiImage,3> &rois);
  bool ensure_init();
  bool ensure_winrt_init(const OcrWorkerConfig &cfg_snapshot, bool need_reinit_snapshot);
  bool ensure_paddle_init(const OcrWorkerConfig &cfg_snapshot, bool need_reinit_snapshot);
  uint32_t infer_winrt_and_match(const std::array<OcrRoiImage,3> &rois, const OcrWorkerConfig &cfg_snapshot);
  uint32_t infer_paddle_and_match(const std::array<OcrRoiImage,3> &rois, const OcrWorkerConfig &cfg_snapshot);
  bool infer_winrt_roi_text(size_t roi_idx, const OcrRoiImage &roi, std::string &text_out);
  void reset_winrt_roi_cache();
  void release_paddle();
  bool load_paddle_dict(const std::string &path);
  static void resize_rgba_to_chw32(const OcrRoiImage &in, int out_h, int &out_w, std::vector<float> &out);
  static void pad_width(std::vector<float> &tensor, int c, int h, int cur_w, int pad_w);

  void *filter_instance_ = nullptr; // used for pd_backfill*
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool running_ = false;
  bool pending_stop_ = false;

  struct Job { uint64_t idx = 0; int back_frames = 90; int hold_frames = 55; std::array<OcrRoiImage,3> rois; };
  std::queue<Job> q_;
  OcrWorkerConfig cfg_;
  OcrWorkerConfig active_cfg_;
  bool need_reinit_ = true;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool apartment_initialized_ = false;
  bool unsupported_logged_ = false;

  std::array<bool, 3> has_roi_cache_ = {false, false, false};
  std::array<uint64_t, 3> last_roi_hash_ = {0, 0, 0};
  std::array<std::string, 3> last_roi_text_;

#if ENABLE_PADDLE_OCR
  std::shared_ptr<paddle_infer::Predictor> predictor_;
#endif
  std::vector<std::string> dict_;
};
