cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

# 使用绝对路径引入 OBS 相关依赖（头文件与库），避免依赖于包配置或导入目标
# 说明：.deps 目录由模板自动拉取 OBS 预编译依赖，此处通过绝对路径显式包含/链接
# 注意：CMAKE_CURRENT_SOURCE_DIR 为绝对路径，确保 include 与 lib 路径为绝对路径
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.deps/include")
# 链接 libobs 及其在 Windows 下所需的 w32-pthreads（均位于 .deps/lib）
# 如后续需要额外系统库，请在此追加（例如 ws2_32 等）
target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.deps/lib/obs.lib"
    "${CMAKE_CURRENT_SOURCE_DIR}/.deps/lib/w32-pthreads.lib"
)

# 如果启用前端 API，同样使用绝对路径链接 obs-frontend-api
if(ENABLE_FRONTEND_API)
  # 头文件仍在 .deps/include 下，无需重复添加
  target_link_libraries(
    ${CMAKE_PROJECT_NAME}
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/.deps/lib/obs-frontend-api.lib"
  )
endif()

# Qt 等后续第三方库将通过 vcpkg 工具链管理（见 CMakePresets.json 中 toolchainFile 设置）
# 在启用 Qt 时仍采用 find_package，实际解析由 vcpkg 提供
if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.cpp)
# 新增：加入延迟滤镜源码
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
  src/filters/predictive_delay_filter.cpp
  src/ocr/ocr_worker.cpp
)

# 链接插件支持静态库（由 helpers_common.cmake 生成）
if(TARGET plugin-support)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plugin-support)
endif()

## Paddle Inference integration (TensorRT disabled by default)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/paddle/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/protobuf/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/glog/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/gflags/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/onednn/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/mklml/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/xxhash/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/utf8proc/include"
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/paddle/lib/paddle_inference.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/protobuf/lib/libprotobuf.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/glog/lib/glog.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/gflags/lib/gflags_static.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/onednn/lib/mkldnn.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/mklml/lib/mklml.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/mklml/lib/libiomp5md.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/xxhash/lib/xxhash.lib"
  "${CMAKE_CURRENT_SOURCE_DIR}/.deps/paddle31/third_party/install/utf8proc/lib/utf8proc_static.lib"
)

if(MSVC)
  # Suppress noisy warnings from Paddle headers on this target only
  target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /wd4251 /wd4244)
endif()

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})

