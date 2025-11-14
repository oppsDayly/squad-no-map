#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// 检测结果结构体
struct DetectionResult {
    float x, y, width, height;  // 边界框坐标 (归一化坐标 0-1)
    float confidence;           // 置信度
    int class_id;              // 类别ID
    std::string class_name;    // 类别名称
};

// ROI图像数据结构 (复用OCR的结构)
struct DetectionRoiImage {
    uint64_t frame_index = 0;
    int w = 0;
    int h = 0;
    int channels = 4; // RGBA
    std::vector<uint8_t> data; // tightly packed RGBA
};

// 目标检测配置
struct ObjectDetectionConfig {
    bool enable = false;
    bool debug_log = false;
    
    // GPU配置
    bool use_cpu = false;
    int gpu_id = 0;
    int gpu_mem_mb = 512;
    
    // 模型配置
    std::string model_path;        // ONNX模型路径
    std::string class_names_path;  // 类别名称文件路径
    
    // 检测参数
    double conf_threshold = 0.5;   // 置信度阈值
    double nms_threshold = 0.4;    // NMS阈值
    int input_width = 640;         // 模型输入宽度
    int input_height = 640;        // 模型输入高度
    
    // 目标类别过滤 (只检测指定类别)
    std::vector<int> target_classes; // 空表示检测所有类别
    
    // 延迟参数
    int back_frames = 90;
    int hold_frames = 55;
    int cpu_threads = 1;
    
    // 推理引擎选择
    enum InferenceEngine {
        ONNX_RUNTIME_DML,    // ONNX Runtime + DirectML (AMD/NVIDIA)
        ONNX_RUNTIME_CUDA,   // ONNX Runtime + CUDA (NVIDIA only)
        OPENCV_DNN_CUDA,     // OpenCV DNN + CUDA (NVIDIA only)
        OPENCV_DNN_CPU       // OpenCV DNN + CPU (fallback)
    } inference_engine = ONNX_RUNTIME_DML;
};

// 前向声明
namespace Ort { class Session; class Env; }
namespace cv { class Net; }

class ObjectDetectionWorker {
public:
    explicit ObjectDetectionWorker(void *filter_instance);
    ~ObjectDetectionWorker();

    void start();
    void stop();
    void update_config(const ObjectDetectionConfig &cfg);
    void submit(uint64_t frame_index, const DetectionRoiImage &roi);

private:
    static constexpr size_t kMaxQueueDepth = 2;
    
    void run();
    uint32_t detect_and_match(const DetectionRoiImage &roi);
    bool ensure_init();
    bool load_class_names(const std::string &path);
    
    // ONNX Runtime相关方法
    bool init_onnx_runtime();
    std::vector<DetectionResult> infer_onnx(const DetectionRoiImage &roi);
    
    // OpenCV DNN相关方法
    bool init_opencv_dnn();
    std::vector<DetectionResult> infer_opencv(const DetectionRoiImage &roi);
    
    // 通用图像预处理
    void preprocess_image(const DetectionRoiImage &input, std::vector<float> &output);
    std::vector<DetectionResult> postprocess_detections(
        const std::vector<float> &outputs, 
        int original_width, 
        int original_height
    );
    
    // NMS后处理
    std::vector<DetectionResult> apply_nms(const std::vector<DetectionResult> &detections);
    
    void *filter_instance_ = nullptr; // used for pd_backfill*
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool running_ = false;
    bool pending_stop_ = false;

    struct Job { 
        uint64_t idx = 0; 
        int back_frames = 90; 
        int hold_frames = 55; 
        DetectionRoiImage roi; 
    };
    std::queue<Job> q_;
    ObjectDetectionConfig cfg_;
    ObjectDetectionConfig active_cfg_;
    bool need_reinit_ = true;

    // ONNX Runtime相关成员
    std::shared_ptr<Ort::Env> ort_env_;
    std::shared_ptr<Ort::Session> ort_session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // OpenCV DNN相关成员
    std::shared_ptr<cv::Net> cv_net_;
    
    // 类别名称
    std::vector<std::string> class_names_;
};