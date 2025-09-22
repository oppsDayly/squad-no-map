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

#include <filesystem>

#include <fstream>

#include "ocr/ocr_worker.h"

#include "filters/predictive_delay_filter.h"

#include "filters/occluder_builtin.h"

#define S_DELAY_MS "pd_delay_ms"

#define S_BACK_FRAMES "pd_back_frames"

#define S_HOLD_FRAMES "pd_hold_frames"

#define S_OCC_PATH "pd_occ_path"

#define S_OCC_AUTO "pd_occ_auto_align"

#define S_SHOW_ROI  "pd_show_roi"

#define S_ROI_THICK "pd_roi_thickness"

#define S_ROI_COLOR "pd_roi_color"

// Built-in ROI rectangles defined in percent of frame dimensions
static constexpr size_t k_roi_count = 3;
static constexpr double k_roi_x_defaults[k_roi_count] = {29.6, 25.8, 64.6};
static constexpr double k_roi_y_defaults[k_roi_count] = {2.4, 12.0, 10.8};
static constexpr double k_roi_w_defaults[k_roi_count] = {4.1, 4.4, 4.1};
static constexpr double k_roi_h_defaults[k_roi_count] = {3.5, 3.5, 3.5};

#define S_OCC_OFFSET_X "pd_occ_offset_x"

#define S_OCC_OFFSET_Y "pd_occ_offset_y"

#define S_OCC_W "pd_occ_w"

#define S_OCC_H "pd_occ_h"

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

struct pd_b_state {

    uint64_t next_index = 0;

    int back_frames = 90;

    int hold_frames = 55;

    uint32_t occ_w = 0;

    uint32_t occ_h = 0;

    int occ_offset_x = 0;

    int occ_offset_y = 0;

    char *occ_path = nullptr;

    gs_image_file_t occ_image = {};

    bool occ_loaded = false;

    bool occ_auto_align = true;

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

static inline bool pd_check_size(struct pd_filter_data *f);

static void pd_check_interval(struct pd_filter_data *f);

static pd_b_state *pd_get_state(void *key);

static size_t pd_num_frames(struct deque *buf);

static inline void pd_state_load_occluder(pd_b_state *s, const char *path);

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

static std::string pd_builtin_occluder_storage_path()
{

    char *module_path = obs_module_config_path("builtin_occluder.jpg");

    if (!module_path) {

        return std::string();

    }

    std::string path = module_path;

    bfree(module_path);

    return path;

}

static bool pd_write_builtin_occluder_file(const std::string &path)
{

    std::error_code ec;

    std::filesystem::path p(path);

    if (!p.parent_path().empty()) {

        std::filesystem::create_directories(p.parent_path(), ec);

        if (ec) {

            obs_log(LOG_ERROR, "[pd][occ] failed to create builtin occluder directory: %s", path.c_str());

            return false;

        }

    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);

    if (!ofs) {

        obs_log(LOG_ERROR, "[pd][occ] failed to open builtin occluder: %s", path.c_str());

        return false;

    }

    ofs.write(reinterpret_cast<const char *>(g_builtin_occluder), static_cast<std::streamsize>(g_builtin_occluder_size));

    if (!ofs) {

        obs_log(LOG_ERROR, "[pd][occ] failed to write builtin occluder: %s", path.c_str());

        return false;

    }

    ofs.close();

    return true;

}

static std::string pd_get_builtin_occluder_path()
{

    std::string path = pd_builtin_occluder_storage_path();

    if (path.empty())

        return std::string();

    if (!pd_write_builtin_occluder_file(path))

        return std::string();

    return path;

}


static void pd_tick(void *data, float t);

static void pd_defaults(obs_data_t *s);

static void pd_draw_occluder_overlay(struct pd_filter_data *f, pd_b_state *st, uint64_t frame_index);

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

    pd_draw_occluder_overlay(f, st, frame.index);

}
static void pd_draw_occluder_overlay(struct pd_filter_data *f, pd_b_state *st, uint64_t frame_index)
{

    if (!st || !st->occ_active)
        return;

    if (frame_index < st->occ_active_from || frame_index > st->occ_active_to)
        return;

    if ((!st->occ_loaded || !st->occ_image.texture) && st->occ_path && *st->occ_path) {
        if (st->debug_log) {
            obs_log(LOG_INFO, "[pd][occ] reload request (loaded=%s, texture=%p)",
                    st->occ_loaded ? "true" : "false", st->occ_image.texture);
        }

        st->occ_loaded = false;
        pd_state_load_occluder(st, st->occ_path);

        if (st->debug_log) {
            obs_log(LOG_INFO, "[pd][occ] reload result (loaded=%s, texture=%p, size=%ux%u)",
                    st->occ_loaded ? "true" : "false", st->occ_image.texture,
                    (unsigned)st->occ_image.cx, (unsigned)st->occ_image.cy);
        }
    }

    gs_blend_state_push();
    gs_enable_blending(true);
    gs_blend_function(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA);
    gs_ortho(0.0f, (float)f->cx, 0.0f, (float)f->cy, -100.0f, 100.0f);

    const bool prev = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(true);

    double roi2_left_pct = (k_roi_count >= 2) ? st->roi_x_pct[1] : 0.0;
    roi2_left_pct = std::clamp(roi2_left_pct, 0.0, 100.0);
    int roi2_left_px = (int)std::lround(roi2_left_pct * 0.01 * (double)f->cx);
    roi2_left_px = std::clamp(roi2_left_px, 0, (int)f->cx);

    int target_w = (int)f->cx - roi2_left_px;
    if (target_w <= 0)
        target_w = (int)f->cx;

    int target_h = (int)f->cy;
    if (target_h <= 0)
        target_h = (st->occ_loaded && st->occ_image.cy > 0) ? (int)st->occ_image.cy : 1;

    int draw_x = roi2_left_px + st->occ_offset_x;
    int draw_y = st->occ_offset_y;

    if ((int)f->cx > 0) {
        if (draw_x < 0)
            draw_x = 0;
        if (draw_x + target_w > (int)f->cx)
            draw_x = std::max(0, (int)f->cx - target_w);
    }

    if ((int)f->cy > 0) {
        if (draw_y < 0)
            draw_y = 0;
        if (draw_y + target_h > (int)f->cy)
            draw_y = std::max(0, (int)f->cy - target_h);
    }

    bool drew_cover = false;

    if (st->occ_loaded && st->occ_image.texture && target_w > 0 && target_h > 0) {
        gs_effect_t *e = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture_srgb(gs_effect_get_param_by_name(e, "image"), st->occ_image.texture);

        gs_matrix_push();
        gs_matrix_identity();
        gs_matrix_translate3f((float)draw_x, (float)draw_y, 0.0f);

        bool drew = false;

        while (gs_effect_loop(e, "Draw")) {
            gs_draw_sprite(st->occ_image.texture, 0, target_w, target_h);
            drew = true;
        }

        gs_matrix_pop();

        drew_cover = drew;

        if (st->debug_log) {
            obs_log(LOG_INFO, "[pd][occ] draw texture=%p dst=(%d,%d,%d,%d)",
                    st->occ_image.texture, draw_x, draw_y, target_w, target_h);
        }

        if (st->debug_log && !drew)
            obs_log(LOG_WARNING, "[pd][occ] effect loop did not draw texture");
    }

    if (!drew_cover && target_w > 0 && target_h > 0) {
        gs_matrix_push();
        gs_matrix_identity();
        gs_matrix_translate3f((float)draw_x, (float)draw_y, 0.0f);

        pd_draw_rect(0.0f, 0.0f, (float)target_w, (float)target_h, 0x000000, 0.85f);

        gs_matrix_pop();
        drew_cover = true;

        if (st->debug_log) {
            obs_log(LOG_INFO, "[pd][occ] fallback rect drawn at (%d,%d,%d,%d)",
                    draw_x, draw_y, target_w, target_h);
        }
    }

    if (!drew_cover && st->debug_log) {
        if (st->occ_path && *st->occ_path)
            obs_log(LOG_WARNING, "[pd][occ] occluder image unavailable: %s", st->occ_path);
        else
            obs_log(LOG_DEBUG, "[pd][occ] occluder draw skipped (no image/rect)");
    }

    if (st->debug_log) {
        obs_log(LOG_INFO, "[pd][occ] overlay %s for frame %llu", drew_cover ? "applied" : "skipped",
                (unsigned long long)frame_index);
    }

    gs_enable_framebuffer_srgb(prev);

    gs_blend_state_pop();
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

    return s;

}

static inline void pd_release_state(void *key)

{

    std::lock_guard<std::mutex> lk(g_b_state_mu);

    auto it = g_b_states.find(key);

    if (it != g_b_states.end()) {

        pd_b_state *s = it->second;

        if (s->occ_path) bfree(s->occ_path);

        if (s->model_dir) bfree(s->model_dir);

        if (s->dict_path) bfree(s->dict_path);

        if (s->occ_loaded) {

            obs_enter_graphics();

            gs_image_file_free(&s->occ_image);

            obs_leave_graphics();

        }

        bfree(s);

        g_b_states.erase(it);

    }

}

static inline void pd_state_load_occluder(pd_b_state *s, const char *path_in)
{

    if (!s)
        return;

    std::string requested_path = path_in ? std::string(path_in) : std::string();

    std::string current_path = s->occ_path ? std::string(s->occ_path) : std::string();

    if (s->occ_loaded && !requested_path.empty() && requested_path == current_path)
        return;

    if (s->occ_loaded) {

        obs_enter_graphics();

        gs_image_file_free(&s->occ_image);

        obs_leave_graphics();

        s->occ_loaded = false;

    }

    if (s->occ_path) { bfree(s->occ_path); s->occ_path = nullptr; }

    if (requested_path.empty())
        return;

    std::string builtin_path = pd_builtin_occluder_storage_path();

    bool using_builtin = false;

    if (!builtin_path.empty() && requested_path == builtin_path) {

        std::string prepared_path = pd_get_builtin_occluder_path();

        if (prepared_path.empty()) {

            obs_log(LOG_ERROR, "[pd][occ] unable to prepare builtin occluder file");

            return;

        }

        requested_path = prepared_path;

        using_builtin = true;

    }

    s->occ_path = bstrdup(requested_path.c_str());

    gs_image_file_init(&s->occ_image, s->occ_path);

    obs_enter_graphics();

    gs_image_file_init_texture(&s->occ_image);

    obs_leave_graphics();

    s->occ_loaded = s->occ_image.loaded && s->occ_image.texture != nullptr;

    obs_log(LOG_INFO, "[pd] occluder %s: %s", s->occ_loaded ? "loaded" : "failed", s->occ_path);

    if (using_builtin) {

        std::error_code rm_ec;

        std::filesystem::remove(requested_path, rm_ec);

        if (rm_ec) {

            obs_log(LOG_WARNING, "[pd][occ] failed to remove builtin occluder file: %s", requested_path.c_str());

        }

    }

}

// External API: schedule backfill from OCR thread (or any caller)

extern "C" void pd_backfill_range(void *filter_instance, unsigned long long from, unsigned long long to)

{

    if (!filter_instance) return;

    pd_b_state *st = pd_get_state(filter_instance);

    st->pending_from = (uint64_t)from;

    st->pending_to = (uint64_t)to;

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

        st->has_pending_cmd = false;

        if (st->debug_log) {

            obs_log(LOG_INFO, "[pd][occ] activate range [%llu, %llu]", (unsigned long long)st->occ_active_from, (unsigned long long)st->occ_active_to);

        }

    }

    if (st->occ_active && st->last_present_index >= st->occ_active_to) {

        st->occ_active = false;

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

    // update occlusion state

    pd_b_state *st = pd_get_state(f);
    pd_apply_roi_defaults(st);

    st->back_frames = (int)obs_data_get_int(s, S_BACK_FRAMES);

    st->hold_frames = (int)obs_data_get_int(s, S_HOLD_FRAMES);

    st->occ_offset_x = (int)obs_data_get_int(s, S_OCC_OFFSET_X);

    st->occ_offset_y = (int)obs_data_get_int(s, S_OCC_OFFSET_Y);

    st->occ_w = (uint32_t)obs_data_get_int(s, S_OCC_W);

    st->occ_h = (uint32_t)obs_data_get_int(s, S_OCC_H);

    st->occ_auto_align = obs_data_get_bool(s, S_OCC_AUTO);

    if (st->occ_path) { bfree(st->occ_path); st->occ_path = nullptr; }

    std::string builtin_path = pd_builtin_occluder_storage_path();

    pd_state_load_occluder(st, builtin_path.empty() ? nullptr : builtin_path.c_str());

    // ROI overlay

    st->show_roi = obs_data_get_bool(s, S_SHOW_ROI);

    st->roi_thickness = (int)obs_data_get_int(s, S_ROI_THICK);

    st->roi_color = (uint32_t)obs_data_get_int(s, S_ROI_COLOR);

    // OCR settings

    st->enable_ocr = obs_data_get_bool(s, S_ENABLE_OCR);

    st->ocr_interval_ms = (int)obs_data_get_int(s, S_OCR_INTERVAL_MS);

    if (st->ocr_interval_ms < 0) st->ocr_interval_ms = 0;

    st->next_ocr_allowed_time_ns = 0;

    st->cpu_threads = (int)obs_data_get_int(s, S_CPU_THREADS);

    if (st->cpu_threads < 1) st->cpu_threads = 1;

    st->gpu_id = (int)obs_data_get_int(s, S_GPU_ID);

    st->gpu_mem_mb = (int)obs_data_get_int(s, S_GPU_MEM);

    st->conf_threshold = obs_data_get_double(s, S_CONF_THR);

    if (st->model_dir) { bfree(st->model_dir); st->model_dir = nullptr; }

    if (st->dict_path) { bfree(st->dict_path); st->dict_path = nullptr; }

    const char *md = obs_data_get_string(s, S_MODEL_DIR);

    const char *dp = obs_data_get_string(s, S_DICT_PATH);

    if (md && *md) st->model_dir = bstrdup(md);

    if (dp && *dp) st->dict_path = bstrdup(dp);

    st->debug_log = obs_data_get_bool(s, S_DEBUG_LOG);

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

    obs_data_set_default_bool(s, S_OCC_AUTO, true);

    obs_data_set_default_int(s, S_OCC_OFFSET_X, 0);

    obs_data_set_default_int(s, S_OCC_OFFSET_Y, 0);

    obs_data_set_default_int(s, S_OCC_W, 0);

    obs_data_set_default_int(s, S_OCC_H, 0);

    obs_data_set_default_bool(s, S_SHOW_ROI, true);

    obs_data_set_default_int(s, S_ROI_THICK, 2);

    obs_data_set_default_int(s, S_ROI_COLOR, 0x00FF00);

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

	obs_properties_add_bool(props, S_OCC_AUTO, "Occluder Auto Align (Right, V-Center)");

	obs_properties_add_int(props, S_OCC_OFFSET_X, "Occluder Offset X (px)", -16384, 16384, 1);

	obs_properties_add_int(props, S_OCC_OFFSET_Y, "Occluder Offset Y (px)", -16384, 16384, 1);

	obs_properties_add_int(props, S_OCC_W, "Occluder Width (px)", 0, 16384, 1);

	obs_properties_add_int(props, S_OCC_H, "Occluder Height (px)", 0, 16384, 1);

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

    // ROI overlay options

    obs_properties_add_bool(props, S_SHOW_ROI, "Show ROI Boxes");

    obs_properties_add_int(props, S_ROI_THICK, "ROI Border Thickness (px)", 1, 20, 1);

    obs_properties_add_color(props, S_ROI_COLOR, "ROI Color");

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

