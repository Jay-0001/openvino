# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_GPU)
    return()
endif()

set (TARGET_NAME "openvino_intel_gpu_plugin")

#added modificn
option(ENABLE_GTPIN_INTEGRATION "Enable local GTPin integration in Intel GPU plugin" OFF)
if(ENABLE_GTPIN_INTEGRATION)
    set(GTPIN_ROOT "" CACHE PATH "Path to GTPin Profilers directory")

     if(NOT EXISTS "${GTPIN_ROOT}")
        message(FATAL_ERROR
            "GTPIN_ROOT does not exist: ${GTPIN_ROOT}")
    endif()
    
    set(GTPIN_INCLUDE_DIR "${GTPIN_ROOT}/Include")
    set(GTPIN_API_INCLUDE_DIR "${GTPIN_ROOT}/Include/api")
    set(GTPIN_GED_INCLUDE_DIR "${GTPIN_ROOT}/Include/ged/intel64")
endif()



if(OV_COMPILER_IS_INTEL_LLVM)
    # For windows we need to disable warning as error option to make FindSYCL.cmake work
    if (WIN32)
        ov_add_compiler_flags(/WX-)
    endif()

    find_package(IntelSYCL REQUIRED)

    if (WIN32)
        ov_add_compiler_flags(/WX)
        ov_add_compiler_flags(/Wno-ignored-attributes)
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-strict-aliasing)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 4267 4244 conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4244)
    # '<': signed/unsigned mismatch
    ov_add_compiler_flags(/wd4018)

    add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)
    # see https://github.com/oneapi-src/oneDNN/issues/2028
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

if(ENABLE_GPU_DEBUG_CAPS)
    add_definitions(-DGPU_DEBUG_CONFIG=1)
    add_definitions(-DENABLE_DEBUG_CAPS=1)
endif()

set(INTEL_GPU_TARGET_OCL_VERSION "300"
  CACHE STRING "Target version of OpenCL which should be used by GPU plugin")

add_definitions(-DCL_TARGET_OPENCL_VERSION=${INTEL_GPU_TARGET_OCL_VERSION})

set(MAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(thirdparty)
include(thirdparty/cmake/rapidjson.cmake)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Werror)
endif()

add_subdirectory(src/runtime)
add_subdirectory(src/kernel_selector)
add_subdirectory(src/graph)

file(GLOB_RECURSE PLUGIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_gpu/plugin/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GPU"
              SOURCES ${PLUGIN_SOURCES}
              DEFAULT_CONFIG ${PLUGIN_DEFAULT_CONFIG}
              VERSION_DEFINES_FOR src/plugin/plugin.cpp)

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_graph openvino::pugixml)

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/)

#Optional GTPin integration added
#include and links cannot be placed before the target 
if(ENABLE_GTPIN_INTEGRATION)
    target_include_directories(${TARGET_NAME} PRIVATE
        "${GTPIN_INCLUDE_DIR}"
        "${GTPIN_API_INCLUDE_DIR}"
        "${GTPIN_GED_INCLUDE_DIR}"
    )

    target_compile_definitions(${TARGET_NAME} PRIVATE
        ENABLE_GTPIN_INTEGRATION
    )
endif()
#end of GTPin integration logic



ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Workaround to avoid warnings during LTO build
if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow"
                                                  LINK_FLAGS_RELWITHDEBINFO "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow")
endif()

ov_build_target_faster(${TARGET_NAME} PCH)

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Failed because of OpenCL
# must be called after all target_link_libraries
# ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
