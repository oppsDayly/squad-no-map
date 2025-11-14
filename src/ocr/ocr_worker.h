#pragma once

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

struct OcrWorkerConfig {
  bool enable = false;
  bool debug_log = false;
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

namespace paddle_infer { class Predictor; }

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
  bool load_dict(const std::string &path);
  static void resize_rgba_to_chw32(const OcrRoiImage &in, int out_h, int &out_w, std::vector<float> &out);
  static void pad_width(std::vector<float> &tensor, int c, int h, int cur_w, int pad_w);

  void *filter_instance_ = nullptr; // used for pd_backfill*
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool running_ = false;
  bool pending_stop_ = false;

  struct Job { uint64_t idx = 0; int back_frames = 90; int hold_frames = 55; int cpu_threads = 1; std::array<OcrRoiImage,3> rois; };
  std::queue<Job> q_;
  OcrWorkerConfig cfg_;
  OcrWorkerConfig active_cfg_;
  bool need_reinit_ = true;

  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::vector<std::string> dict_;
};
