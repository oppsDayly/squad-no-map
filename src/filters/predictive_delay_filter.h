#pragma once

// 预测遮挡延迟滤镜（仅实现延迟，后续将接入文字识别与提前遮挡）
// 说明：本滤镜复用 OBS 内置 gpu-delay 的思路：将每帧渲染到纹理，再使用环形队列实现 N 毫秒的输出延迟。
// 这里为后续“在识别到目标文字前 N 帧即开始渲染遮挡”搭建基础设施。

#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

// 返回延迟滤镜的 obs_source_info 指针，用于注册
const struct obs_source_info *get_predictive_delay_filter_info(void);

// 外部接口（供 OCR 通知回填遮挡）
// 参数 filter_instance 为滤镜实例指针（即 obs_source_t* 的私有 data 指针）
void pd_backfill_range(void *filter_instance, unsigned long long from, unsigned long long to, uint32_t roi_mask);
void pd_backfill_now(void *filter_instance, int back_frames, int hold_frames);

#ifdef __cplusplus
}
#endif

