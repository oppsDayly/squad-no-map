/*

预测遮挡延迟滤镜（第一阶段：仅实现延迟，不做识别）

- 思路：参考 obs-filters/gpu-delay.c 的帧缓存队列方案，将 target 的画面渲染进循环纹理队列，实现毫秒级延迟。

- 日志：在关键生命周期与渲染路径打点，便于调试。

*/

#include <obs-module.h>

#include <graphics/graphics.h>

#include <graphics/image-file.h>

#include <util/deque.h>

#include <plugin-support.h>

#include <mutex>

#include <unordered_map>

#include <vector>

#include <array>

#include <algorithm>

#include <chrono>

#include <cmath>

#include <cctype>

#include <filesystem>

#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

#include "ocr/ocr_worker.h"

#include "filters/predictive_delay_filter.h"
#include "filters/occluder_builtin.h"

#include <string>

enum class pd_occluder_mode {
    Image = 0,
    Mosaic = 1,
    GaussianBlur = 2,
};

#define S_DELAY_MS "pd_delay_ms"

#define S_BACK_FRAMES "pd_back_frames"

#define S_HOLD_FRAMES "pd_hold_frames"


#define S_SHOW_ROI  "pd_show_roi"

#define S_ROI_THICK "pd_roi_thickness"

#define S_ROI_COLOR "pd_roi_color"
#define S_OCC_MODE  "pd_occ_mode"
#define S_OCC_BORDER "pd_show_occ_border"
#define S_OCC_MOSAIC "pd_occ_mosaic_px"
#define S_OCC_GAUSS  "pd_occ_gauss_strength"

// Convert UTF-8 strings from OBS (which always uses UTF-8) into native filesystem paths.
static std::filesystem::path pd_utf8_to_path(const std::string &utf8)
{
#if defined(_WIN32)
	if (utf8.empty())
		return {};
	int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
	if (wlen <= 0)
		return {};
	std::wstring buffer;
	buffer.resize((size_t)wlen);
	int written = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, buffer.data(), wlen);
	if (written <= 0)
		return {};
	buffer.resize((size_t)written - 1);
	return std::filesystem::path(buffer);
#else
	return std::filesystem::path(utf8);
#endif
}

// Built-in ROI rectangles defined in percent of frame dimensions
static constexpr size_t k_roi_count = 3;
static constexpr double k_roi_x_defaults[k_roi_count] = {29.6, 25.8, 64.6};
static constexpr double k_roi_y_defaults[k_roi_count] = {2.4, 12.0, 10.8};
static constexpr double k_roi_w_defaults[k_roi_count] = {4.1, 4.4, 4.1};
static constexpr double k_roi_h_defaults[k_roi_count] = {3.5, 3.5, 3.5};
static constexpr double k_occ_x_defaults[k_roi_count] = {29.10, 47.60, 53.00};
static constexpr double k_occ_y_defaults[k_roi_count] = {7.30, 15.90, 9.30};
static constexpr double k_occ_w_defaults[k_roi_count] = {70.00, 51.40, 47.00};
static constexpr double k_occ_h_defaults[k_roi_count] = {90.80, 76.80, 81.50};
static constexpr int k_default_mosaic_block_px = 24;
static constexpr int k_default_gaussian_strength = 6;
static constexpr uint32_t k_watermark_color = 0xFFFFFF;
static constexpr float k_watermark_alpha = 0.55f;
static constexpr const char *k_watermark_text = "GAUSS BLUR";
static constexpr uint32_t k_roi1_bit = 1u << 0;
static constexpr uint32_t k_roi2_bit = 1u << 1;
static constexpr uint32_t k_roi3_bit = 1u << 2;
static constexpr uint32_t k_roi12_mask = k_roi1_bit | k_roi2_bit;
static constexpr uint32_t k_roi3_mask = k_roi3_bit;


#define S_ENABLE_OCR "pd_enable_ocr"

#define S_OCR_INTERVAL_MS "pd_ocr_interval_ms"

#define S_GPU_ID     "pd_gpu_id"

#define S_GPU_MEM    "pd_gpu_mem_mb"

#define S_CPU_THREADS "pd_cpu_threads"

#define S_MODEL_DIR  "pd_model_dir"

#define S_CONF_THR   "pd_conf_threshold"

#define S_DICT_PATH  "pd_dict_path"

#define S_DEBUG_LOG  "pd_debug_log"

#define S_USE_CPU    "pd_use_cpu"

#define T_FILTER_NAME obs_module_text("PredictiveDelayFilter")

#define T_DELAY_MS obs_module_text("DelayMs")

struct pd_frame {

	gs_texrender_t *render;

	enum gs_color_space space;

	uint64_t ts;

	uint64_t index; // 单调自增帧索引（用于回写遮挡命令定位）

};

struct pd_filter_data {

	obs_source_t *context;              // 滤镜自身上下文

	struct deque frames;                // 帧环形缓冲区（存放 pd_frame）

	uint64_t delay_ns = 0;              // 期望延迟（纳秒）

	uint64_t interval_ns = 0;           // 视频帧间隔（纳秒）

	uint32_t cx = 0;                    // 目标宽

	uint32_t cy = 0;                    // 目标高

	bool target_valid = false;          // 目标是否有效

	bool processed_frame = false;       // 本 tick 是否已处理帧

};

// -------------------- Retroactive occlusion (Scheme B) state --------------------

struct pd_occ_region {
    double x_pct = 0.0;
    double y_pct = 0.0;
    double w_pct = 0.0;
    double h_pct = 0.0;
    pd_occluder_mode mode = pd_occluder_mode::Mosaic;
    int mosaic_block_px = k_default_mosaic_block_px;
    int gaussian_strength = k_default_gaussian_strength;
    char *image_path = nullptr;
    gs_image_file_t image = {};
    bool image_loaded = false;
};

struct pd_b_state {

    uint64_t next_index = 0;

    int back_frames = 90;

    int hold_frames = 55;

    pd_occ_region occ_regions[k_roi_count] = {};

    uint32_t occ_pending_roi_mask = 0;

    uint32_t occ_active_roi_mask = 0;

    bool has_pending_cmd = false;

    uint64_t pending_from = 0;

    uint64_t pending_to = 0;

    bool occ_active = false;

    uint64_t occ_active_from = 0;

    uint64_t occ_active_to = 0;

    uint64_t last_present_index = 0;

    double roi_x_pct[k_roi_count] = {};

    double roi_y_pct[k_roi_count] = {};

    double roi_w_pct[k_roi_count] = {};

    double roi_h_pct[k_roi_count] = {};

    bool show_roi = true;
    bool show_occ_border = false;

    int roi_thickness = 2;

    uint32_t roi_color = 0x00FF00;

    bool enable_ocr = false;

    int gpu_id = 0;

    int gpu_mem_mb = 512;

    int cpu_threads = 1;

    double conf_threshold = 0.7;

    char *model_dir = nullptr;

    char *dict_path = nullptr;

    bool debug_log = false;

    bool use_cpu = false;

    int ocr_interval_ms = 100;

    uint64_t next_ocr_allowed_time_ns = 0;

};

static inline void pd_apply_roi_defaults(pd_b_state *st)
{

    if (!st) return;

    for (size_t i = 0; i < k_roi_count; ++i) {
        st->roi_x_pct[i] = k_roi_x_defaults[i];
        st->roi_y_pct[i] = k_roi_y_defaults[i];
        st->roi_w_pct[i] = k_roi_w_defaults[i];
        st->roi_h_pct[i] = k_roi_h_defaults[i];
    }
}

static inline void pd_apply_occ_defaults(pd_b_state *st)
{
    if (!st) return;
    for (size_t i = 0; i < k_roi_count; ++i) {
        st->occ_regions[i].x_pct = k_occ_x_defaults[i];
        st->occ_regions[i].y_pct = k_occ_y_defaults[i];
        st->occ_regions[i].w_pct = k_occ_w_defaults[i];
        st->occ_regions[i].h_pct = k_occ_h_defaults[i];
        st->occ_regions[i].mode = pd_occluder_mode::Mosaic;
        st->occ_regions[i].mosaic_block_px = k_default_mosaic_block_px;
        st->occ_regions[i].gaussian_strength = k_default_gaussian_strength;
    }
}

static inline void pd_draw_rect(float x, float y, float w, float h, uint32_t rgb, float alpha);

static void pd_draw_front(struct pd_filter_data *f, pd_b_state *st);
struct pd_occ_rect;

static void pd_draw_roi_boxes(struct pd_filter_data *f, pd_b_state *st);

static void pd_draw_occ_boxes(struct pd_filter_data *f, pd_b_state *st);

static void pd_draw_occluder_overlay(struct pd_filter_data *f, pd_b_state *st, uint64_t frame_index,
                                     gs_texture_t *frame_tex, enum gs_color_space frame_space);

struct occ_res;
static inline occ_res *occ_get(void *key);
static inline void occ_release(void *key);
static inline void occ_ensure(occ_res *res, size_t idx, uint32_t w, uint32_t h);

static void pd_draw_watermark_text(const pd_occ_rect &rect);

struct pd_occ_rect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

static inline double pd_sanitize_pct(double value)
{
    if (!std::isfinite(value))
        return 0.0;
    return std::clamp(value, 0.0, 100.0);
}

static bool pd_compute_occ_rect(const pd_occ_region &region, uint32_t frame_w, uint32_t frame_h, pd_occ_rect &out)
{
    if (!frame_w || !frame_h)
        return false;

    double x_pct = pd_sanitize_pct(region.x_pct);
    double y_pct = pd_sanitize_pct(region.y_pct);
    double w_pct = pd_sanitize_pct(region.w_pct);
    double h_pct = pd_sanitize_pct(region.h_pct);

    double max_w_pct = std::max(0.0, 100.0 - x_pct);
    double max_h_pct = std::max(0.0, 100.0 - y_pct);
    w_pct = std::clamp(w_pct, 0.0, max_w_pct);
    h_pct = std::clamp(h_pct, 0.0, max_h_pct);

    int x = (int)std::lround(x_pct * 0.01 * (double)frame_w);
    int y = (int)std::lround(y_pct * 0.01 * (double)frame_h);
    int w = (int)std::lround(w_pct * 0.01 * (double)frame_w);
    int h = (int)std::lround(h_pct * 0.01 * (double)frame_h);

    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= (int)frame_w || y >= (int)frame_h)
        return false;

    if (w <= 0 || h <= 0)
        return false;

    if (x + w > (int)frame_w)
        w = (int)frame_w - x;
    if (y + h > (int)frame_h)
        h = (int)frame_h - y;

    if (w <= 0 || h <= 0)
        return false;

    out.x = x;
    out.y = y;
    out.w = w;
    out.h = h;
    return true;
}

static void pd_occ_region_release_image(pd_occ_region &region)
{
    if (region.image.texture || region.image_loaded) {
        obs_enter_graphics();
        gs_image_file_free(&region.image);
        obs_leave_graphics();
    }
    region.image = {};
    region.image_loaded = false;
}

static void pd_occ_region_set_image_path(pd_occ_region &region, const char *path)
{
    if (region.image_path) {
        bfree(region.image_path);
        region.image_path = nullptr;
    }
    if (path && *path)
        region.image_path = bstrdup(path);
}

static void pd_occ_region_load_image(pd_occ_region &region, size_t idx, bool debug_log)
{
    pd_occ_region_release_image(region);
    if (!region.image_path || !*region.image_path)
        return;

    gs_image_file_init(&region.image, region.image_path);
    obs_enter_graphics();
    gs_image_file_init_texture(&region.image);
    obs_leave_graphics();
    region.image_loaded = region.image.loaded && region.image.texture != nullptr;
    if (!region.image_loaded) {
        obs_log(LOG_WARNING, "[pd][occ] failed to load image for ROI%zu: %s",
                idx + 1, region.image_path);
        return;
    }
    if (debug_log) {
        obs_log(LOG_INFO, "[pd][occ] loaded image for ROI%zu: %s (%ux%u)",
                idx + 1, region.image_path, (unsigned)region.image.cx, (unsigned)region.image.cy);
    }
}

static std::string pd_extract_builtin_image(size_t idx)
{
    if (!k_builtin_occluder_count)
        return {};
    size_t actual = (idx < k_builtin_occluder_count) ? idx : (k_builtin_occluder_count - 1);
    const auto &def = k_builtin_occluders[actual];
    char *utf8 = obs_module_file(def.filename);
    if (!utf8) {
        obs_log(LOG_ERROR, "[pd][occ] failed to resolve builtin image path for idx=%zu", idx);
        return {};
    }
    std::string path = utf8;
    bfree(utf8);

    const std::filesystem::path native = pd_utf8_to_path(path);
    const std::filesystem::path parent = native.parent_path();
    std::error_code ec;
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            obs_log(LOG_ERROR, "[pd][occ] failed to create directory for %s", path.c_str());
            return {};
        }
    }

    std::ofstream ofs(native, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        obs_log(LOG_ERROR, "[pd][occ] failed to open builtin image for write: %s", path.c_str());
        return {};
    }
    ofs.write(reinterpret_cast<const char *>(def.data), static_cast<std::streamsize>(def.size));
    if (!ofs) {
        obs_log(LOG_ERROR, "[pd][occ] failed to write builtin image: %s", path.c_str());
        return {};
    }
    return path;
}

static gs_samplerstate_t *g_sampler_point = nullptr;
static gs_samplerstate_t *g_sampler_linear = nullptr;
static std::mutex g_sampler_mu;

static gs_samplerstate_t *pd_get_sampler(bool point)
{
    std::lock_guard<std::mutex> lk(g_sampler_mu);
    gs_samplerstate_t **slot = point ? &g_sampler_point : &g_sampler_linear;
    if (*slot)
        return *slot;

    struct gs_sampler_info info = {};
    info.filter = point ? GS_FILTER_POINT : GS_FILTER_LINEAR;
    info.address_u = GS_ADDRESS_CLAMP;
    info.address_v = GS_ADDRESS_CLAMP;
    info.address_w = GS_ADDRESS_CLAMP;

    obs_enter_graphics();
    *slot = gs_samplerstate_create(&info);
    obs_leave_graphics();
    return *slot;
}

static void pd_release_samplers()
{
    std::lock_guard<std::mutex> lk(g_sampler_mu);
    if (g_sampler_point || g_sampler_linear) {
        obs_enter_graphics();
        if (g_sampler_point) {
            gs_samplerstate_destroy(g_sampler_point);
            g_sampler_point = nullptr;
        }
        if (g_sampler_linear) {
            gs_samplerstate_destroy(g_sampler_linear);
            g_sampler_linear = nullptr;
        }
        obs_leave_graphics();
    }
}

struct occ_res {
    gs_texrender_t *downsample[k_roi_count] = {nullptr, nullptr, nullptr};
    uint32_t w[k_roi_count] = {0,0,0};
    uint32_t h[k_roi_count] = {0,0,0};
};

static std::mutex g_occ_mu;
static std::unordered_map<void*, occ_res*> g_occ_map;

static inline occ_res *occ_get(void *key)
{
    std::lock_guard<std::mutex> lk(g_occ_mu);
    auto it = g_occ_map.find(key);
    if (it != g_occ_map.end())
        return it->second;
    occ_res *res = (occ_res *)bzalloc(sizeof(occ_res));
    g_occ_map[key] = res;
    return res;
}

static inline void occ_release(void *key)
{
    std::lock_guard<std::mutex> lk(g_occ_mu);
    auto it = g_occ_map.find(key);
    if (it == g_occ_map.end())
        return;
    occ_res *res = it->second;
    obs_enter_graphics();
    for (size_t i = 0; i < k_roi_count; ++i) {
        if (res->downsample[i])
            gs_texrender_destroy(res->downsample[i]);
    }
    obs_leave_graphics();
    bfree(res);
    g_occ_map.erase(it);
}

static inline void occ_ensure(occ_res *res, size_t idx, uint32_t w, uint32_t h)
{
    if (!res || idx >= k_roi_count || w == 0 || h == 0)
        return;
    bool need_new = (!res->downsample[idx]) || res->w[idx] != w || res->h[idx] != h;
    if (!need_new)
        return;
    obs_enter_graphics();
    if (res->downsample[idx])
        gs_texrender_destroy(res->downsample[idx]);
    res->downsample[idx] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    res->w[idx] = w;
    res->h[idx] = h;
    obs_leave_graphics();
}

static bool pd_capture_roi_to_target(gs_texrender_t *target, uint32_t target_w, uint32_t target_h,
                                     enum gs_color_space space, gs_texture_t *frame_tex,
                                     const pd_occ_rect &rect, uint32_t frame_w, uint32_t frame_h)
{
    if (!target || !frame_tex || target_w == 0 || target_h == 0 || rect.w <= 0 || rect.h <= 0)
        return false;

    gs_texrender_reset(target);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    bool ok = false;
    if (gs_texrender_begin_with_color_space(target, target_w, target_h, space)) {
        struct vec4 clear_color; vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
        gs_ortho(0.0f, (float)rect.w, 0.0f, (float)rect.h, -100.0f, 100.0f);

        gs_matrix_push();
        gs_matrix_identity();
        gs_matrix_translate3f(-(float)rect.x, -(float)rect.y, 0.0f);

        gs_effect_t *copy_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture_srgb(gs_effect_get_param_by_name(copy_effect, "image"), frame_tex);
        while (gs_effect_loop(copy_effect, "Draw")) {
            gs_draw_sprite(frame_tex, 0, frame_w, frame_h);
        }
        gs_matrix_pop();

        gs_texrender_end(target);
        ok = true;
    }
    gs_blend_state_pop();
    return ok;
}

static bool pd_draw_downsampled_region(struct pd_filter_data *f, gs_texture_t *frame_tex,
                                       enum gs_color_space frame_space, const pd_occ_rect &rect,
                                       occ_res *gfx, size_t roi_index, uint32_t down_w, uint32_t down_h,
                                       bool point_sample)
{
    if (!f || !frame_tex || !gfx || roi_index >= k_roi_count)
        return false;
    if (down_w == 0 || down_h == 0)
        return false;

    occ_ensure(gfx, roi_index, down_w, down_h);
    gs_texrender_t *tmp = gfx->downsample[roi_index];
    if (!tmp)
        return false;

    if (!pd_capture_roi_to_target(tmp, down_w, down_h, frame_space, frame_tex, rect, f->cx, f->cy))
        return false;

    gs_texture_t *down_tex = gs_texrender_get_texture(tmp);
    if (!down_tex)
        return false;

    gs_effect_t *draw_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_eparam_t *image_param = gs_effect_get_param_by_name(draw_effect, "image");
    if (point_sample) {
        if (auto *sampler = pd_get_sampler(true))
            gs_effect_set_next_sampler(image_param, sampler);
    } else {
        if (auto *sampler = pd_get_sampler(false))
            gs_effect_set_next_sampler(image_param, sampler);
    }
    gs_effect_set_texture_srgb(image_param, down_tex);

    gs_matrix_push();
    gs_matrix_identity();
    gs_matrix_translate3f((float)rect.x, (float)rect.y, 0.0f);
    gs_matrix_scale3f((float)rect.w / (float)down_w, (float)rect.h / (float)down_h, 1.0f);

    bool drew = false;
    while (gs_effect_loop(draw_effect, "Draw")) {
        gs_draw_sprite(down_tex, 0, down_w, down_h);
        drew = true;
    }
    gs_matrix_pop();
    return drew;
}

static inline bool pd_check_size(struct pd_filter_data *f);

static void pd_check_interval(struct pd_filter_data *f);

static pd_b_state *pd_get_state(void *key);

static size_t pd_num_frames(struct deque *buf);

static size_t pd_num_frames(struct deque *buf)

{

    return buf->size / sizeof(struct pd_frame);

}

static const char *pd_get_name(void *unused);

static const char *pd_get_name(void *unused)

{

    UNUSED_PARAMETER(unused);

    return T_FILTER_NAME;

}

static void pd_tick(void *data, float t);

static void pd_defaults(obs_data_t *s);


static void pd_draw_front(struct pd_filter_data *f, pd_b_state *st)
{

    if (!f || !f->context)
        return;

    if (!f->frames.size) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    struct pd_frame frame;
    deque_peek_front(&f->frames, &frame, sizeof(frame));

    gs_texture_t *tex = gs_texrender_get_texture(frame.render);
    if (!tex) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    gs_ortho(0.0f, (float)f->cx, 0.0f, (float)f->cy, -100.0f, 100.0f);

    const bool prev_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(true);

    gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_effect_set_texture_srgb(gs_effect_get_param_by_name(effect, "image"), tex);

    gs_matrix_push();
    gs_matrix_identity();

    while (gs_effect_loop(effect, "Draw")) {
        gs_draw_sprite(tex, 0, 0, 0);
    }

    gs_matrix_pop();

    gs_enable_framebuffer_srgb(prev_srgb);
    gs_blend_state_pop();

    if (st)
        st->last_present_index = frame.index;

    pd_draw_occluder_overlay(f, st, frame.index, tex, frame.space);
}


static void pd_draw_occluder_overlay(struct pd_filter_data *f, pd_b_state *st, uint64_t frame_index, gs_texture_t *frame_tex, enum gs_color_space frame_space)
{

    if (!st || !st->occ_active)
        return;

    if (frame_index < st->occ_active_from || frame_index > st->occ_active_to)
        return;

    if (!frame_tex)
        return;

    gs_blend_state_push();
    gs_enable_blending(true);
    gs_blend_function(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA);
    gs_ortho(0.0f, (float)f->cx, 0.0f, (float)f->cy, -100.0f, 100.0f);

    const bool prev = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(true);

    const int frame_w = (int)f->cx;
    const int frame_h = (int)f->cy;

    uint32_t active_mask = st->occ_active_roi_mask;
    if (!active_mask) {
        gs_enable_framebuffer_srgb(prev);
        gs_blend_state_pop();
        return;
    }

    occ_res *gfx = occ_get(f);

    for (size_t roi = 0; roi < k_roi_count; ++roi) {
        if ((active_mask & (1u << roi)) == 0)
            continue;

        pd_occ_rect rect;
        if (!pd_compute_occ_rect(st->occ_regions[roi], frame_w, frame_h, rect))
            continue;

        const pd_occ_region &region = st->occ_regions[roi];
        bool drew = false;

        switch (region.mode) {
        case pd_occluder_mode::Image:
            if (region.image_loaded && region.image.texture) {
                const float tex_w = (float)region.image.cx;
                const float tex_h = (float)region.image.cy;
                if (tex_w > 0.0f && tex_h > 0.0f) {
                    float scale = std::min((float)rect.w / tex_w, (float)rect.h / tex_h);
                    if (!std::isfinite(scale) || scale <= 0.0f)
                        scale = 1.0f;
                    float draw_w = tex_w * scale;
                    float draw_h = tex_h * scale;
                    if (draw_w <= 0.0f || draw_h <= 0.0f)
                        break;
                    float offset_x = (float)rect.x + ((float)rect.w - draw_w) * 0.5f;
                    float offset_y = (float)rect.y + ((float)rect.h - draw_h) * 0.5f;

                    gs_effect_t *e = obs_get_base_effect(OBS_EFFECT_DEFAULT);
                    gs_effect_set_texture_srgb(gs_effect_get_param_by_name(e, "image"), region.image.texture);

                    gs_matrix_push();
                    gs_matrix_identity();
                    gs_matrix_translate3f(offset_x, offset_y, 0.0f);

                    while (gs_effect_loop(e, "Draw")) {
                        gs_draw_sprite(region.image.texture, 0,
                                       (uint32_t)std::round(draw_w),
                                       (uint32_t)std::round(draw_h));
                        drew = true;
                    }

                    gs_matrix_pop();

                    if (st->debug_log) {
                        obs_log(LOG_INFO, "[pd][occ] roi%zu image cover drawn at (%d,%d,%d,%d)",
                                roi + 1, rect.x, rect.y, rect.w, rect.h);
                    }
                }
            }
            break;
        case pd_occluder_mode::Mosaic: {
            int block = region.mosaic_block_px <= 0 ? 1 : region.mosaic_block_px;
            uint32_t down_w = (uint32_t)std::max(1, (rect.w + block - 1) / block);
            uint32_t down_h = (uint32_t)std::max(1, (rect.h + block - 1) / block);
            down_w = std::min<uint32_t>(down_w, (uint32_t)rect.w);
            down_h = std::min<uint32_t>(down_h, (uint32_t)rect.h);
            drew = pd_draw_downsampled_region(f, frame_tex, frame_space, rect, gfx, roi, down_w, down_h, true);
            if (st->debug_log) {
                obs_log(LOG_INFO, "[pd][occ] roi%zu mosaic drawn at (%d,%d,%d,%d) block=%d",
                        roi + 1, rect.x, rect.y, rect.w, rect.h, block);
            }
            break;
        }
        case pd_occluder_mode::GaussianBlur: {
            int strength = std::max(1, region.gaussian_strength);
            uint32_t down_w = std::max<uint32_t>(1u, (uint32_t)rect.w / (uint32_t)strength);
            uint32_t down_h = std::max<uint32_t>(1u, (uint32_t)rect.h / (uint32_t)strength);
            drew = pd_draw_downsampled_region(f, frame_tex, frame_space, rect, gfx, roi, down_w, down_h, false);
            if (drew)
                pd_draw_watermark_text(rect);
            if (st->debug_log) {
                obs_log(LOG_INFO, "[pd][occ] roi%zu gaussian blur drawn at (%d,%d,%d,%d) strength=%d",
                        roi + 1, rect.x, rect.y, rect.w, rect.h, strength);
            }
            break;
        }
        default:
            break;
        }

        if (!drew) {
            gs_matrix_push();
            gs_matrix_identity();
            gs_matrix_translate3f((float)rect.x, (float)rect.y, 0.0f);
            pd_draw_rect(0.0f, 0.0f, (float)rect.w, (float)rect.h, 0x000000, 0.85f);
            gs_matrix_pop();
            if (st->debug_log) {
                obs_log(LOG_INFO, "[pd][occ] roi%zu fallback rect drawn at (%d,%d,%d,%d)",
                        roi + 1, rect.x, rect.y, rect.w, rect.h);
            }
        }

    }

    gs_enable_framebuffer_srgb(prev);

    gs_blend_state_pop();
}



static inline void pd_draw_rect(float x, float y, float w, float h, uint32_t rgb, float alpha)
{

    if (w <= 0.0f || h <= 0.0f || alpha <= 0.0f)
        return;

    gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);

    if (!solid)
        return;

    const float inv255 = 1.0f / 255.0f;

    struct vec4 color;

    color.x = ((float)((rgb >> 16) & 0xFF)) * inv255;
    color.y = ((float)((rgb >> 8) & 0xFF)) * inv255;
    color.z = ((float)(rgb & 0xFF)) * inv255;
    color.w = alpha;

    gs_effect_set_vec4(gs_effect_get_param_by_name(solid, "color"), &color);

    gs_matrix_push();
    gs_matrix_translate3f(x, y, 0.0f);

    uint32_t draw_w = (uint32_t)std::round(w);
    uint32_t draw_h = (uint32_t)std::round(h);

    if (!draw_w)
        draw_w = 1;

    if (!draw_h)
        draw_h = 1;

    while (gs_effect_loop(solid, "Solid")) {
        gs_draw_sprite(nullptr, 0, draw_w, draw_h);
    }

    gs_matrix_pop();

}

static void pd_draw_roi_boxes(struct pd_filter_data *f, pd_b_state *st)
{

    if (!f || !st || !st->show_roi)
        return;

    const int thickness = std::max(1, st->roi_thickness);
    const uint32_t color = st->roi_color;
    const float alpha = 1.0f;

    for (size_t i = 0; i < k_roi_count; ++i) {
        double xp_pct = st->roi_x_pct[i];
        double yp_pct = st->roi_y_pct[i];
        double wp_pct = st->roi_w_pct[i];
        double hp_pct = st->roi_h_pct[i];

        if (wp_pct <= 0.0 || hp_pct <= 0.0)
            continue;

        double xp = std::clamp(xp_pct, 0.0, 100.0) * 0.01;
        double yp = std::clamp(yp_pct, 0.0, 100.0) * 0.01;
        double wp = std::clamp(wp_pct, 0.0, 100.0) * 0.01;
        double hp = std::clamp(hp_pct, 0.0, 100.0) * 0.01;

        float rx = (float)(xp * (double)f->cx);
        float ry = (float)(yp * (double)f->cy);
        float rw = (float)(wp * (double)f->cx);
        float rh = (float)(hp * (double)f->cy);

        if (rw <= 0.0f || rh <= 0.0f)
            continue;

        if (rx < 0.0f)
            rx = 0.0f;
        if (ry < 0.0f)
            ry = 0.0f;
        if (rx + rw > (float)f->cx)
            rw = (float)f->cx - rx;
        if (ry + rh > (float)f->cy)
            rh = (float)f->cy - ry;

        if (rw <= 0.0f || rh <= 0.0f)
            continue;

        float t = (float)thickness;
        if (t > rw)
            t = rw;
        if (t > rh)
            t = rh;

        pd_draw_rect(rx, ry, rw, t, color, alpha);
        pd_draw_rect(rx, ry + rh - t, rw, t, color, alpha);
        pd_draw_rect(rx, ry, t, rh, color, alpha);
        pd_draw_rect(rx + rw - t, ry, t, rh, color, alpha);
    }

}

static void pd_draw_occ_boxes(struct pd_filter_data *f, pd_b_state *st)
{
    if (!f || !st || !st->show_occ_border)
        return;
    if (f->cx == 0 || f->cy == 0)
        return;

    const int thickness = std::max(1, st->roi_thickness);
    const uint32_t color = st->roi_color;
    const float alpha = 1.0f;

    for (size_t i = 0; i < k_roi_count; ++i) {
        pd_occ_rect rect;
        if (!pd_compute_occ_rect(st->occ_regions[i], f->cx, f->cy, rect))
            continue;

        float t = (float)thickness;
        float x = (float)rect.x;
        float y = (float)rect.y;
        float w = (float)rect.w;
        float h = (float)rect.h;

        if (t > w) t = w;
        if (t > h) t = h;

        pd_draw_rect(x, y, w, t, color, alpha);
        pd_draw_rect(x, y + h - t, w, t, color, alpha);
        pd_draw_rect(x, y, t, h, color, alpha);
        pd_draw_rect(x + w - t, y, t, h, color, alpha);
    }
}





static std::mutex g_b_state_mu;

static std::unordered_map<void*, pd_b_state*> g_b_states;

static inline pd_b_state *pd_get_state(void *key)

{

    std::lock_guard<std::mutex> lk(g_b_state_mu);

    auto it = g_b_states.find(key);

    if (it != g_b_states.end()) return it->second;

    pd_b_state *s = (pd_b_state*)bzalloc(sizeof(pd_b_state));

    g_b_states[key] = s;
    pd_apply_roi_defaults(s);
    pd_apply_occ_defaults(s);

    return s;

}

static inline void pd_release_state(void *key)

{

    std::lock_guard<std::mutex> lk(g_b_state_mu);

    auto it = g_b_states.find(key);

    if (it != g_b_states.end()) {

        pd_b_state *s = it->second;

        for (size_t i = 0; i < k_roi_count; ++i) {
            pd_occ_region &region = s->occ_regions[i];
            pd_occ_region_release_image(region);
            if (region.image_path) {
                bfree(region.image_path);
                region.image_path = nullptr;
            }
        }

        if (s->model_dir) bfree(s->model_dir);

        if (s->dict_path) bfree(s->dict_path);

        bfree(s);

        g_b_states.erase(it);
        if (g_b_states.empty())
            pd_release_samplers();

    }

}



// External API: schedule backfill from OCR thread (or any caller)

extern "C" void pd_backfill_range(void *filter_instance, unsigned long long from, unsigned long long to, uint32_t roi_mask)

{

    if (!filter_instance) return;

    pd_b_state *st = pd_get_state(filter_instance);

    st->pending_from = (uint64_t)from;

    st->pending_to = (uint64_t)to;

    st->occ_pending_roi_mask = roi_mask;

    st->has_pending_cmd = true;

}

extern "C" void pd_backfill_now(void *filter_instance, int back_frames, int hold_frames)

{

    if (!filter_instance) return;

    pd_b_state *st = pd_get_state(filter_instance);

    uint64_t cur = st->next_index;

    uint64_t from = (cur > (uint64_t)back_frames) ? (cur - (uint64_t)back_frames) : 0ULL;

    uint64_t to = cur + (uint64_t)hold_frames;

    st->pending_from = from;

    st->pending_to = to;

    st->occ_pending_roi_mask = k_roi12_mask;

    st->has_pending_cmd = true;

}

// -------------------- ROI capture and OCR submission --------------------

struct roi_res {

    gs_texrender_t *rend[3] = {nullptr, nullptr, nullptr};

    gs_stagesurf_t *stage[3] = {nullptr, nullptr, nullptr};

    uint32_t w[3] = {0,0,0};

    uint32_t h[3] = {0,0,0};

    // 回退路径需要：全帧staging表面及其尺寸

    gs_stagesurf_t *full_stage = nullptr; // 当texrender begin失败时，从整帧纹理stage后在CPU裁剪

    uint32_t full_w = 0;                  // 全帧宽

    uint32_t full_h = 0;                  // 全帧高

};

static std::mutex g_roi_mu;

static std::unordered_map<void*, roi_res*> g_roi_map;

static inline roi_res *roi_get(void *key)

{

    std::lock_guard<std::mutex> lk(g_roi_mu);

    auto it = g_roi_map.find(key);

    if (it != g_roi_map.end()) return it->second;

    roi_res *r = (roi_res*)bzalloc(sizeof(roi_res));

    g_roi_map[key] = r;

    return r;

}

static inline void roi_release(void *key)

{

    std::lock_guard<std::mutex> lk(g_roi_mu);

    auto it = g_roi_map.find(key);

    if (it == g_roi_map.end()) return;

    roi_res *r = it->second;

    obs_enter_graphics();

    for (int i = 0; i < 3; ++i) {

        if (r->rend[i]) gs_texrender_destroy(r->rend[i]);

        if (r->stage[i]) gs_stagesurface_destroy(r->stage[i]);

    }

    // 释放全帧回退stagesurface

    if (r->full_stage) gs_stagesurface_destroy(r->full_stage);

    obs_leave_graphics();

    bfree(r);

    g_roi_map.erase(it);

}

static inline void roi_ensure(roi_res *rr, int idx, uint32_t w, uint32_t h)

{

    if (w == 0 || h == 0) return;

    bool need_new = (!rr->rend[idx]) || rr->w[idx] != w || rr->h[idx] != h || !rr->stage[idx];

    if (!need_new) return;

    obs_enter_graphics();

    if (rr->rend[idx]) gs_texrender_destroy(rr->rend[idx]);

    if (rr->stage[idx]) gs_stagesurface_destroy(rr->stage[idx]);

    // Force RGBA8 for staging safety

    rr->rend[idx] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);

    rr->stage[idx] = gs_stagesurface_create(w, h, GS_RGBA);

    rr->w[idx] = w; rr->h[idx] = h;

    obs_leave_graphics();

}

// 新增：确保全帧回退stagesurface存在且尺寸匹配

static inline void roi_full_stage_ensure(roi_res *rr, uint32_t w, uint32_t h)

{

    if (w == 0 || h == 0) return;

    bool need_new = (!rr->full_stage) || rr->full_w != w || rr->full_h != h;

    if (!need_new) return;

    obs_enter_graphics();

    if (rr->full_stage) gs_stagesurface_destroy(rr->full_stage);

    rr->full_stage = gs_stagesurface_create(w, h, GS_RGBA);

    rr->full_w = w; rr->full_h = h;

    obs_leave_graphics();

}

static inline void pd_capture_and_submit(struct pd_filter_data *f, pd_b_state *st, gs_texture_t *frame_tex, enum gs_color_space space, uint64_t frame_index)

{

    if (!st->enable_ocr) return;

    extern OcrWorker *pd_worker_get(void *key);

    OcrWorker *worker = pd_worker_get(f);

    if (!worker) return;

    OcrWorkerConfig cfg;

    cfg.enable = st->enable_ocr;

    cfg.use_cpu = st->use_cpu;

    cfg.gpu_id = st->gpu_id;

    cfg.gpu_mem_mb = st->gpu_mem_mb;

    if (st->model_dir) cfg.model_dir = st->model_dir;

    if (st->dict_path) cfg.dict_path = st->dict_path;

    cfg.conf_threshold = st->conf_threshold;

    cfg.debug_log = st->debug_log;

    cfg.back_frames = st->back_frames;

    cfg.hold_frames = st->hold_frames;

    cfg.cpu_threads = st->use_cpu ? 1 : st->cpu_threads;

    worker->update_config(cfg);

    if (!frame_tex) return;

    uint64_t now_ns = 0;

    if (st->ocr_interval_ms > 0) {

        auto now = std::chrono::steady_clock::now();

        now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

        if (st->next_ocr_allowed_time_ns != 0 && now_ns < st->next_ocr_allowed_time_ns) {

            return;

        }

    }

    roi_res *rr = roi_get(f);

    std::array<OcrRoiImage,3> rois;

    for (size_t i = 0; i < k_roi_count; ++i) {

        double xp_pct = st->roi_x_pct[i];

        double yp_pct = st->roi_y_pct[i];

        double wp_pct = st->roi_w_pct[i];

        double hp_pct = st->roi_h_pct[i];

        if (wp_pct <= 0.0 || hp_pct <= 0.0) { rois[i] = {}; continue; }

        double xp = xp_pct * 0.01;

        double yp = yp_pct * 0.01;

        double wp = wp_pct * 0.01;

        double hp = hp_pct * 0.01;

        uint32_t rx = (uint32_t)(xp * (double)f->cx + 0.5);

        uint32_t ry = (uint32_t)(yp * (double)f->cy + 0.5);

        uint32_t rw = (uint32_t)(wp * (double)f->cx + 0.5);

        uint32_t rh = (uint32_t)(hp * (double)f->cy + 0.5);

        if (rw == 0 || rh == 0) { rois[i] = {}; continue; }

        if (rx + rw > f->cx) rw = f->cx - rx;

        if (ry + rh > f->cy) rh = f->cy - ry;

        roi_ensure(rr, (int)i, rw, rh);

        if (st->debug_log && (frame_index % 60 == 0)) {

            obs_log(LOG_INFO, "[pd][ocr] ROI%d rect=(%u,%u,%u,%u)", (int)i + 1, rx, ry, rw, rh);

        }

        gs_blend_state_push();

        gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

        // 在begin前reset一次，避免上一次失败导致状态异常

        if (rr->rend[i]) gs_texrender_reset(rr->rend[i]);

        // 使用上游给定的颜色空间进行FBO配置

        if (gs_texrender_begin_with_color_space(rr->rend[i], rw, rh, space)) {

             struct vec4 clear_color; vec4_zero(&clear_color);

             gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);

             gs_ortho(0.0f, (float)rw, 0.0f, (float)rh, -100.0f, 100.0f);

             const bool prev = gs_framebuffer_srgb_enabled();

             // 统一关闭SRGB写入，避免采样/写入Gamma差异影响结果

             gs_enable_framebuffer_srgb(false);

             gs_effect_t *e = obs_get_base_effect(OBS_EFFECT_DEFAULT);

             // 使用SRGB绑定有利于匹配默认Draw技术的期望

             gs_effect_set_texture_srgb(gs_effect_get_param_by_name(e, "image"), frame_tex);

             // 调试：记录一次绘制是否发生以及纹理尺寸

             static int roi_draw_log_count = 0;

             bool drew = false;

             if (st->debug_log && roi_draw_log_count < 20) {

                 uint32_t ftw = gs_texture_get_width(frame_tex);

                 uint32_t fth = gs_texture_get_height(frame_tex);

                 obs_log(LOG_INFO, "[pd][ocr] ROI%d draw begin ok, frame_tex=%ux%u -> roi=%ux%u, rx=%u ry=%u", i+1, ftw, fth, rw, rh, rx, ry);

             }

             // 使用平移将原始大纹理的ROI区域绘制到小FBO

             uint32_t ftw = gs_texture_get_width(frame_tex);

             uint32_t fth = gs_texture_get_height(frame_tex);

             gs_matrix_push();

             gs_matrix_translate3f(- (float)rx, - (float)ry, 0.0f);

             while (gs_effect_loop(e, "Draw")) {

                 // 按实际纹理尺寸去绘制，避免与源尺寸不一致

                 gs_draw_sprite(frame_tex, 0, (uint32_t)ftw, (uint32_t)fth);

                 drew = true;

             }

             gs_matrix_pop();

             if (st->debug_log && roi_draw_log_count < 20) {

                 obs_log(LOG_INFO, "[pd][ocr] ROI%d draw finished, drew=%s", i+1, drew?"true":"false");

                 roi_draw_log_count++;

             }

             gs_enable_framebuffer_srgb(prev);

             gs_texrender_end(rr->rend[i]);

         } else {

             if (st->debug_log) {

                 obs_log(LOG_WARNING, "[pd][ocr] ROI%d gs_texrender_begin_with_color_space FAILED (w=%u h=%u space=%d)", i+1, rw, rh, (int)space);

             }

             // 回退路径：直接从整帧纹理stage到CPU并裁剪ROI

             roi_full_stage_ensure(rr, f->cx, f->cy);

             if (rr->full_stage) {

                 gs_stage_texture(rr->full_stage, frame_tex);

                 uint8_t *fdata = nullptr; uint32_t fls = 0;

                 if (gs_stagesurface_map(rr->full_stage, &fdata, &fls)) {

                     OcrRoiImage img; img.frame_index = frame_index; img.w = (int)rw; img.h = (int)rh; img.channels = 4;

                     img.data.resize((size_t)rw * (size_t)rh * 4);

                     for (uint32_t row = 0; row < rh; ++row) {

                         const uint8_t *src = fdata + (size_t)(ry + row) * (size_t)fls + (size_t)rx * 4;

                         uint8_t *dst = img.data.data() + (size_t)row * (size_t)rw * 4;

                         memcpy(dst, src, (size_t)rw * 4);

                     }

                     gs_stagesurface_unmap(rr->full_stage);

                     rois[i] = std::move(img);

                     // 打印一次回退生效日志

                     if (st->debug_log) {

                         obs_log(LOG_WARNING, "[pd][ocr] ROI%d fallback staged from full frame", i+1);

                     }

                 } else {

                     rois[i] = {};

                 }

             }

         }

        gs_blend_state_pop();

         // Stage to CPU memory（仅当texrender绘制成功时才从ROI纹理拷贝）

         gs_texture_t *roi_tex = gs_texrender_get_texture(rr->rend[i]);

         if (!roi_tex) {

             // 如果之前已经通过回退路径填充了rois[i]，则跳过

             if (rois[i].data.empty()) { rois[i] = {}; }

             continue;

         }

         // 如果rois[i]已通过回退路径填充，则无需再次stage

         if (!rois[i].data.empty()) continue;

         gs_stage_texture(rr->stage[i], roi_tex);

         uint8_t *data = nullptr; uint32_t linesize = 0;

         if (gs_stagesurface_map(rr->stage[i], &data, &linesize)) {

             OcrRoiImage img;

             img.frame_index = frame_index;

             img.w = (int)rw; img.h = (int)rh; img.channels = 4;

             img.data.resize((size_t)rh * (size_t)rw * 4);

             // copy row by row into tightly packed RGBA

             for (uint32_t row = 0; row < rh; ++row) {

                 memcpy(img.data.data() + (size_t)row * (size_t)rw * 4,

                        data + (size_t)row * (size_t)linesize,

                        (size_t)rw * 4);

             }

             gs_stagesurface_unmap(rr->stage[i]);

            // 调试：前若干次无条件打印 ROI 原始 RGBA 均值，便于快速确认是否为全黑

            static int roi_log_count = 0;

            if (st->debug_log && roi_log_count < 20) {

                // 打印原始RGBA的均值，帮助确认是否拿到了全0

                double mr = 0, mg = 0, mb = 0, ma = 0; size_t pix = (size_t)rw * (size_t)rh;

                for (size_t p = 0; p < pix; ++p) {

                    const uint8_t *px = &img.data[p * 4];

                    mr += px[0]; mg += px[1]; mb += px[2]; ma += px[3];

                }

                mr /= (double)pix; mg /= (double)pix; mb /= (double)pix; ma /= (double)pix;

                const uint8_t *px0 = img.data.data();

                obs_log(LOG_INFO, "[pd][ocr] ROI%d RGBA mean=(%.1f,%.1f,%.1f,%.1f) firstpx=(%u,%u,%u,%u) size=%ux%u linesize=%u",

                        i+1, mr, mg, mb, ma, (unsigned)px0[0], (unsigned)px0[1], (unsigned)px0[2], (unsigned)px0[3], rw, rh, linesize);

                roi_log_count++;

            }

             rois[i] = std::move(img);

         } else {

             rois[i] = {};

         }

    }

    // Submit to worker

    if (st->ocr_interval_ms > 0) {

        uint64_t interval_ns = (uint64_t)st->ocr_interval_ms * 1000000ULL;

        st->next_ocr_allowed_time_ns = now_ns + interval_ns;

    } else {

        st->next_ocr_allowed_time_ns = 0;

    }

    worker->submit(frame_index, rois);

}

// Global worker registry

static std::mutex g_worker_mu;

static std::unordered_map<void*, OcrWorker*> g_workers;

OcrWorker *pd_worker_get(void *key)

{

    std::lock_guard<std::mutex> lk(g_worker_mu);

    auto it = g_workers.find(key);

    if (it != g_workers.end()) return it->second;

    OcrWorker *w = new OcrWorker(key);

    w->start();

    g_workers[key] = w;

    return w;

}

static void pd_worker_release(void *key)

{

    std::lock_guard<std::mutex> lk(g_worker_mu);

    auto it = g_workers.find(key);

    if (it != g_workers.end()) {

        OcrWorker *w = it->second;

        w->stop();

        delete w;

        g_workers.erase(it);

    }

}

static void pd_tick(void *data, float t)

{

    UNUSED_PARAMETER(t);

    auto *f = (struct pd_filter_data *)data;

    if (!f)

        return;

    pd_check_size(f);

    if (f->target_valid)

        pd_check_interval(f);

    f->processed_frame = false;

    pd_b_state *st = pd_get_state(f);

    if (!st)

        return;

    if (st->has_pending_cmd) {

        st->occ_active = true;

        st->occ_active_from = st->pending_from;

        st->occ_active_to = st->pending_to;

        st->occ_active_roi_mask = st->occ_pending_roi_mask ? st->occ_pending_roi_mask : k_roi12_mask;

        st->occ_pending_roi_mask = 0;

        st->has_pending_cmd = false;

        if (st->debug_log) {

            obs_log(LOG_INFO, "[pd][occ] activate range [%llu, %llu] mask=0x%X",
                    (unsigned long long)st->occ_active_from, (unsigned long long)st->occ_active_to,
                    st->occ_active_roi_mask);

        }

    }

    if (st->occ_active && st->last_present_index >= st->occ_active_to) {

        st->occ_active = false;

        st->occ_active_roi_mask = 0;

        if (st->debug_log) {

            obs_log(LOG_INFO, "[pd][occ] deactivate after frame %llu", (unsigned long long)st->last_present_index);

        }

    }

}

static void pd_free_textures(struct pd_filter_data *f)

{

	obs_log(LOG_DEBUG, "[pd] free_textures, frames=%zu", pd_num_frames(&f->frames));

	obs_enter_graphics();

	while (f->frames.size) {

		struct pd_frame frame;

		deque_pop_front(&f->frames, &frame, sizeof(frame));

		gs_texrender_destroy(frame.render);

	}

	deque_free(&f->frames);

	obs_leave_graphics();

}

static void pd_update_interval(struct pd_filter_data *f, uint64_t new_interval_ns)

{

	if (!f->target_valid) {

		pd_free_textures(f);

		return;

	}

	f->interval_ns = new_interval_ns;

	size_t need = (size_t)(f->delay_ns / new_interval_ns);

	size_t have = pd_num_frames(&f->frames);

	obs_log(LOG_INFO, "[pd] update_interval: interval_ns=%llu, need=%zu, have=%zu", (unsigned long long)new_interval_ns, need, have);

	if (need > have) {

		obs_enter_graphics();

		deque_upsize(&f->frames, need * sizeof(struct pd_frame));

		for (size_t i = have; i < need; ++i) {

			struct pd_frame *frame = (struct pd_frame *)deque_data(&f->frames, i * sizeof(*frame));

			frame->render = gs_texrender_create(GS_RGBA, GS_ZS_NONE);

			frame->index = 0;

		}

		obs_leave_graphics();

	} else if (need < have) {

		obs_enter_graphics();

		while (pd_num_frames(&f->frames) > need) {

			struct pd_frame frame;

			deque_pop_front(&f->frames, &frame, sizeof(frame));

			gs_texrender_destroy(frame.render);

		}

		obs_leave_graphics();

	}

}

static inline void pd_check_interval(struct pd_filter_data *f)

{

	struct obs_video_info ovi = {0};

	obs_get_video_info(&ovi);

	uint64_t interval_ns = util_mul_div64(ovi.fps_den, 1000000000ULL, ovi.fps_num);

	if (interval_ns != f->interval_ns)

		pd_update_interval(f, interval_ns);

}

static inline void pd_reset_textures(struct pd_filter_data *f)

{

	f->interval_ns = 0;

	pd_free_textures(f);

	pd_check_interval(f);

}

static inline bool pd_check_size(struct pd_filter_data *f)

{

	obs_source_t *target = obs_filter_get_target(f->context);

	f->target_valid = !!target;

	if (!f->target_valid) return true;

	uint32_t cx = obs_source_get_base_width(target);

	uint32_t cy = obs_source_get_base_height(target);

	f->target_valid = !!cx && !!cy;

	if (!f->target_valid) return true;

	if (cx != f->cx || cy != f->cy) {

		obs_log(LOG_INFO, "[pd] target size change: %ux%u -> %ux%u", f->cx, f->cy, cx, cy);

		f->cx = cx; f->cy = cy; pd_reset_textures(f); return true;

	}

	return false;

}

static void pd_update(void *data, obs_data_t *s)

{

    struct pd_filter_data *f = (struct pd_filter_data *)data;

    f->delay_ns = (uint64_t)obs_data_get_int(s, S_DELAY_MS) * 1000000ULL; // ms -> ns

    obs_log(LOG_INFO, "[pd] update: delay_ms=%lld", (long long)obs_data_get_int(s, S_DELAY_MS));

    pd_b_state *st = pd_get_state(f);

    st->back_frames = (int)obs_data_get_int(s, S_BACK_FRAMES);
    st->hold_frames = (int)obs_data_get_int(s, S_HOLD_FRAMES);
    st->occ_pending_roi_mask = 0;
    st->occ_active_roi_mask = 0;
    st->occ_active = false;
    st->occ_active_from = 0;
    st->occ_active_to = 0;
    st->has_pending_cmd = false;
    st->pending_from = 0;
    st->pending_to = 0;

    st->show_roi = obs_data_get_bool(s, S_SHOW_ROI);
    st->roi_thickness = (int)obs_data_get_int(s, S_ROI_THICK);
    st->roi_color = (uint32_t)obs_data_get_int(s, S_ROI_COLOR);
    st->show_occ_border = obs_data_get_bool(s, S_OCC_BORDER);

    st->debug_log = obs_data_get_bool(s, S_DEBUG_LOG);

    int mode_val = (int)obs_data_get_int(s, S_OCC_MODE);
    if (mode_val < (int)pd_occluder_mode::Image || mode_val > (int)pd_occluder_mode::GaussianBlur)
        mode_val = (int)pd_occluder_mode::Mosaic;
    pd_occluder_mode occ_mode = static_cast<pd_occluder_mode>(mode_val);

    int mosaic_global = (int)obs_data_get_int(s, S_OCC_MOSAIC);
    if (mosaic_global < 1)
        mosaic_global = 1;
    int gauss_global = (int)obs_data_get_int(s, S_OCC_GAUSS);
    if (gauss_global < 1)
        gauss_global = 1;

    for (size_t i = 0; i < k_roi_count; ++i) {
        st->roi_x_pct[i] = k_roi_x_defaults[i];
        st->roi_y_pct[i] = k_roi_y_defaults[i];
        st->roi_w_pct[i] = k_roi_w_defaults[i];
        st->roi_h_pct[i] = k_roi_h_defaults[i];

        pd_occ_region &region = st->occ_regions[i];
        region.x_pct = k_occ_x_defaults[i];
        region.y_pct = k_occ_y_defaults[i];
        region.w_pct = k_occ_w_defaults[i];
        region.h_pct = k_occ_h_defaults[i];

        region.mode = occ_mode;
        region.mosaic_block_px = mosaic_global;
        region.gaussian_strength = gauss_global;

        std::string builtin_path = pd_extract_builtin_image(i);
        if (!builtin_path.empty())
            pd_occ_region_set_image_path(region, builtin_path.c_str());
        else
            pd_occ_region_set_image_path(region, nullptr);

        if (region.mode == pd_occluder_mode::Image)
            pd_occ_region_load_image(region, i, st->debug_log);
        else
            pd_occ_region_release_image(region);
    }

    st->enable_ocr = obs_data_get_bool(s, S_ENABLE_OCR);
    st->ocr_interval_ms = (int)obs_data_get_int(s, S_OCR_INTERVAL_MS);
    if (st->ocr_interval_ms < 0)
        st->ocr_interval_ms = 0;
    st->next_ocr_allowed_time_ns = 0;

    st->cpu_threads = (int)obs_data_get_int(s, S_CPU_THREADS);
    if (st->cpu_threads < 1)
        st->cpu_threads = 1;
    st->gpu_id = (int)obs_data_get_int(s, S_GPU_ID);
    st->gpu_mem_mb = (int)obs_data_get_int(s, S_GPU_MEM);
    st->conf_threshold = obs_data_get_double(s, S_CONF_THR);

    if (st->model_dir) { bfree(st->model_dir); st->model_dir = nullptr; }
    if (st->dict_path) { bfree(st->dict_path); st->dict_path = nullptr; }

    const char *md = obs_data_get_string(s, S_MODEL_DIR);
    const char *dp = obs_data_get_string(s, S_DICT_PATH);

    if (md && *md) st->model_dir = bstrdup(md);
    if (dp && *dp) st->dict_path = bstrdup(dp);

    st->use_cpu = obs_data_get_bool(s, S_USE_CPU);
    if (st->use_cpu)
        st->cpu_threads = 1;

    // full reset

    f->cx = f->cy = 0; f->interval_ns = 0;

    pd_free_textures(f);

}

static bool pd_test_trigger_clicked(obs_properties_t *, obs_property_t *, void *priv)

{

    auto *f = (pd_filter_data *)priv;

    pd_b_state *st = pd_get_state(f);

    uint64_t cur = st->next_index;

    uint64_t from = (cur > (uint64_t)st->back_frames) ? (cur - (uint64_t)st->back_frames) : 0ULL;

    uint64_t to = cur + (uint64_t)st->hold_frames;

    st->pending_from = from;

    st->pending_to = to;

    st->has_pending_cmd = true;

    obs_log(LOG_INFO, "[pd] test trigger: backfill [%llu, %llu]", (unsigned long long)from, (unsigned long long)to);

    return true;

}

static void pd_defaults(obs_data_t *s)

{

    obs_data_set_default_int(s, S_DELAY_MS, 300);

    obs_data_set_default_int(s, S_BACK_FRAMES, 90);

    obs_data_set_default_int(s, S_HOLD_FRAMES, 55);

    obs_data_set_default_bool(s, S_SHOW_ROI, true);

    obs_data_set_default_int(s, S_ROI_THICK, 2);

    obs_data_set_default_int(s, S_ROI_COLOR, 0x00FF00);
    obs_data_set_default_bool(s, S_OCC_BORDER, false);
    obs_data_set_default_int(s, S_OCC_MODE, (int)pd_occluder_mode::Mosaic);
    obs_data_set_default_int(s, S_OCC_MOSAIC, k_default_mosaic_block_px);
    obs_data_set_default_int(s, S_OCC_GAUSS, k_default_gaussian_strength);

    obs_data_set_default_bool(s, S_ENABLE_OCR, false);

    obs_data_set_default_int(s, S_OCR_INTERVAL_MS, 100);

    obs_data_set_default_bool(s, S_USE_CPU, false);

    obs_data_set_default_int(s, S_CPU_THREADS, 1);

    obs_data_set_default_int(s, S_GPU_ID, 0);

    obs_data_set_default_int(s, S_GPU_MEM, 512);

    obs_data_set_default_double(s, S_CONF_THR, 0.7);

    obs_data_set_default_bool(s, S_DEBUG_LOG, false);

}

static obs_properties_t *pd_properties(void *data)

{

	UNUSED_PARAMETER(data);

	obs_properties_t *props = obs_properties_create();

	obs_property_t *p = obs_properties_add_int(props, S_DELAY_MS, T_DELAY_MS, 0, 5000, 1);

	obs_property_int_set_suffix(p, " ms");

	obs_properties_add_int(props, S_BACK_FRAMES, "Back Frames", 0, 300, 1);

	obs_properties_add_int(props, S_HOLD_FRAMES, "Hold Frames", 0, 300, 1);

    // OCR settings

    obs_properties_add_bool(props, S_ENABLE_OCR, "Enable OCR (PP-OCRv4 rec)");

    obs_properties_add_int(props, S_OCR_INTERVAL_MS, "OCR Interval (ms, 0 = every frame)", 0, 10000, 10);

    obs_properties_add_bool(props, S_USE_CPU, "Use CPU Inference (disable CUDA/TRT)");

    obs_properties_add_int(props, S_CPU_THREADS, "CPU Threads", 1, 16, 1);

    obs_properties_add_int(props, S_GPU_ID, "GPU Device ID", 0, 7, 1);

    obs_properties_add_int(props, S_GPU_MEM, "GPU Memory MB", 256, 16384, 64);

    obs_properties_add_path(props, S_MODEL_DIR, "Paddle Rec Model Dir", OBS_PATH_DIRECTORY, NULL, NULL);

    obs_properties_add_float(props, S_CONF_THR, "Text Confidence Threshold", 0.0, 1.0, 0.01);

    obs_properties_add_path(props, S_DICT_PATH, "Character Dict (ppocr_keys_v1.txt)", OBS_PATH_FILE, NULL, NULL);

    obs_properties_add_bool(props, S_DEBUG_LOG, "Verbose OCR Debug Log");

    // Occluder mode (global)
    obs_property_t *occ_mode = obs_properties_add_list(props, S_OCC_MODE, "Occluder Mode",
                                                       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(occ_mode, "Image", (int)pd_occluder_mode::Image);
    obs_property_list_add_int(occ_mode, "Mosaic", (int)pd_occluder_mode::Mosaic);
    obs_property_list_add_int(occ_mode, "Gaussian Blur", (int)pd_occluder_mode::GaussianBlur);

    obs_properties_add_int(props, S_OCC_MOSAIC, "Occluder Mosaic Block (px)", 1, 512, 1);
    obs_properties_add_int(props, S_OCC_GAUSS, "Occluder Gaussian Strength", 1, 64, 1);

    // ROI overlay options

    obs_properties_add_bool(props, S_SHOW_ROI, "Show ROI Boxes");

    obs_properties_add_int(props, S_ROI_THICK, "ROI Border Thickness (px)", 1, 20, 1);

    obs_properties_add_color(props, S_ROI_COLOR, "ROI Color");
    obs_properties_add_bool(props, S_OCC_BORDER, "Show Occluder Borders");

    obs_properties_add_button(props, "pd_test_trigger", "Trigger Backfill Now", pd_test_trigger_clicked);

    return props;

}

static void *pd_create(obs_data_t *settings, obs_source_t *context)

{

	struct pd_filter_data *f = (struct pd_filter_data *)bzalloc(sizeof(*f));

	f->context = context;

	deque_init(&f->frames); // 初始化环形缓冲

    obs_log(LOG_INFO, "[pd] create");

    (void)pd_get_state(f);

	obs_source_update(context, settings);

	return f;

}

static void pd_destroy(void *data)

{

    struct pd_filter_data *f = (struct pd_filter_data *)data;

    obs_log(LOG_INFO, "[pd] destroy");

    pd_free_textures(f);

    pd_release_state(f);

    roi_release(f);
    occ_release(f);

    pd_worker_release(f);

    bfree(f);

}

static void pd_render(void *data, gs_effect_t *effect)

{

	UNUSED_PARAMETER(effect);

	struct pd_filter_data *f = (struct pd_filter_data *)data;

	obs_source_t *target = obs_filter_get_target(f->context);

	obs_source_t *parent = obs_filter_get_parent(f->context);

	if (!f->target_valid || !target || !parent) { obs_source_skip_video_filter(f->context); return; }

    pd_b_state *st = pd_get_state(f);

	// 计算需要的帧数（根据 delay 和 interval）

	size_t cur = pd_num_frames(&f->frames);

	size_t need = (f->interval_ns > 0) ? (size_t)(f->delay_ns / f->interval_ns) : 0;

	if (need == 0) { // 不需要延迟，直接透传

		obs_source_skip_video_filter(f->context);

		return;

	}

	// 已经在本 tick 处理过帧：直接绘制当前队首

	if (f->processed_frame) { pd_draw_front(f, st); return; }

	// 获取目标色彩空间与纹理格式

	const enum gs_color_space preferred[] = {GS_CS_SRGB, GS_CS_SRGB_16F, GS_CS_709_EXTENDED};

	const enum gs_color_space space = obs_source_get_color_space(target, OBS_COUNTOF(preferred), preferred);

	const enum gs_color_format format = gs_get_format_from_space(space);

	// 如果当前队列长度尚未达到 need，执行“暖机”：仅采集一帧入队并透传

	if (cur < need) {

		struct pd_frame frame = {};

		frame.render = gs_texrender_create(format, GS_ZS_NONE);

		gs_texrender_reset(frame.render);

		gs_blend_state_push();

		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		if (gs_texrender_begin_with_color_space(frame.render, f->cx, f->cy, space)) {

			uint32_t parent_flags = obs_source_get_output_flags(target);

			bool custom_draw = (parent_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;

			bool async = (parent_flags & OBS_SOURCE_ASYNC) != 0;

			struct vec4 clear_color; vec4_zero(&clear_color);

			gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);

			gs_ortho(0.0f, (float)f->cx, 0.0f, (float)f->cy, -100.0f, 100.0f);

			if (target == parent && !custom_draw && !async) obs_source_default_render(target); else obs_source_video_render(target);

			gs_texrender_end(frame.render);

			frame.space = space;

		}

		gs_blend_state_pop();

		frame.index = st->next_index++;

		deque_push_back(&f->frames, &frame, sizeof(frame));

			f->processed_frame = true;

		obs_log(LOG_DEBUG, "[pd] warmup: cur=%zu need=%zu", cur + 1, need);

		// 暖机阶段直接透传原画面

		obs_source_skip_video_filter(f->context);

		return;

	}

	// 达到目标长度：弹出最旧帧，写入当前帧，再绘制新的队首，实现延迟

	struct pd_frame frame; 

	deque_pop_front(&f->frames, &frame, sizeof(frame));

	if (gs_texrender_get_format(frame.render) != format) { gs_texrender_destroy(frame.render); frame.render = gs_texrender_create(format, GS_ZS_NONE); }

	gs_texrender_reset(frame.render);

	gs_blend_state_push();

	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

	if (gs_texrender_begin_with_color_space(frame.render, f->cx, f->cy, space)) {

		uint32_t parent_flags = obs_source_get_output_flags(target);

		bool custom_draw = (parent_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;

		bool async = (parent_flags & OBS_SOURCE_ASYNC) != 0;

		struct vec4 clear_color; vec4_zero(&clear_color);

		gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);

		gs_ortho(0.0f, (float)f->cx, 0.0f, (float)f->cy, -100.0f, 100.0f);

		if (target == parent && !custom_draw && !async) obs_source_default_render(target); else obs_source_video_render(target);

		gs_texrender_end(frame.render);

		frame.space = space;

	}

	gs_blend_state_pop();

	frame.index = st->next_index++;

	deque_push_back(&f->frames, &frame, sizeof(frame));

    pd_draw_front(f, st);

    pd_draw_roi_boxes(f, st);
    pd_draw_occ_boxes(f, st);

    f->processed_frame = true;

	// Capture ROIs from current frame and submit for OCR

	pd_capture_and_submit(f, st, gs_texrender_get_texture(frame.render), space, frame.index);

}

static enum gs_color_space pd_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces)

{

	struct pd_filter_data *f = (struct pd_filter_data *)data;

	obs_source_t *target = obs_filter_get_target(f->context);

	obs_source_t *parent = obs_filter_get_parent(f->context);

	if (!f->target_valid || !target || !parent || !f->frames.size) return (count > 0) ? preferred_spaces[0] : GS_CS_SRGB;

	struct pd_frame frame; deque_peek_front(&f->frames, &frame, sizeof(frame));

	enum gs_color_space space = frame.space;

	for (size_t i = 0; i < count; ++i) { space = preferred_spaces[i]; if (space == frame.space) break; }

	return space;

}

// 结尾：以函数返回 info，逐字段赋值避免 C++20 指定初始化

extern "C" const struct obs_source_info *get_predictive_delay_filter_info(void)

{

    static struct obs_source_info info; // 静态存储期，确保生命周期覆盖注册与使用

    // 仅初始化一次

    if (info.id == nullptr) {

        info.id = "predictive_delay";

        info.type = OBS_SOURCE_TYPE_FILTER;

        info.output_flags = OBS_SOURCE_VIDEO;

        info.get_name = pd_get_name;

        info.create = pd_create;

        info.destroy = pd_destroy;

        info.update = pd_update;

        info.get_defaults = pd_defaults;

        info.get_properties = pd_properties;

        info.video_tick = pd_tick;

        info.video_render = pd_render;

        info.video_get_color_space = pd_get_color_space;

    }

    return &info;

}



struct pd_glyph {
    uint8_t rows[7];
    int width;
};

static pd_glyph pd_lookup_glyph(char c)
{
    switch (c) {
    case 'A': return {{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, 5};
    case 'B': return {{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, 5};
    case 'G': return {{0x0E,0x11,0x10,0x17,0x11,0x11,0x0E}, 5};
    case 'L': return {{0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, 5};
    case 'R': return {{0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, 5};
    case 'S': return {{0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, 5};
    case 'U': return {{0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, 5};
    case ' ': return {{0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 3};
    default:  return {{0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 3};
    }
}

static void pd_draw_watermark_text(const pd_occ_rect &rect)
{
    if (rect.w <= 0 || rect.h <= 0)
        return;
    float unit = std::max(1.0f, std::min((float)rect.w, (float)rect.h) / 140.0f);
    float glyph_height = 7.0f * unit;
    const float margin = 6.0f;
    const float base_x = (float)rect.x + margin;
    float base_y = (float)rect.y + (float)rect.h - glyph_height - margin;
    if (base_y < (float)rect.y)
        base_y = (float)rect.y;
    float cursor = base_x;
    for (const char *p = k_watermark_text; p && *p; ++p) {
        pd_glyph glyph = pd_lookup_glyph((char)toupper((unsigned char)*p));
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = glyph.rows[row];
            for (int col = 0; col < glyph.width; ++col) {
                if (bits & (1u << (glyph.width - col - 1))) {
                    float x = cursor + col * unit;
                    float y = base_y + row * unit;
                    pd_draw_rect(x, y, unit, unit, k_watermark_color, k_watermark_alpha);
                }
            }
        }
        cursor += (float)glyph.width * unit + unit;
    }
}
