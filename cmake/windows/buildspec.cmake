# CMake Windows build dependencies module

include_guard(GLOBAL)

include(buildspec_common)

# _check_dependencies_windows: Set up Windows slice for _check_dependencies
function(_check_dependencies_windows)
  set(arch ${CMAKE_VS_PLATFORM_NAME})
  set(platform windows-${arch})

  set(dependencies_dir "${CMAKE_CURRENT_SOURCE_DIR}/.deps")
  set(prebuilt_filename "windows-deps-VERSION-ARCH-REVISION.zip")
  set(prebuilt_destination "obs-deps-VERSION-ARCH")
  set(qt6_filename "windows-deps-qt6-VERSION-ARCH-REVISION.zip")
  set(qt6_destination "obs-deps-qt6-VERSION-ARCH")
  set(_obs_dev_files_present FALSE)
  if(EXISTS "${dependencies_dir}/include/obs-module.h" AND EXISTS "${dependencies_dir}/lib/obs.lib")
    set(_obs_dev_files_present TRUE)
  endif()

  set(dependencies_list prebuilt)

  if(ENABLE_QT OR NOT _obs_dev_files_present)
    list(APPEND dependencies_list qt6)
  endif()

  if(_obs_dev_files_present)
    message(STATUS "OBS development files found in .deps; skipping obs-studio source setup")
  else()
    message(STATUS "OBS development files missing; setting up obs-studio source dependency")
    list(APPEND dependencies_list obs-studio)
  endif()

  _check_dependencies()
endfunction()

_check_dependencies_windows()
