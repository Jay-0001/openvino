cmake_minimum_required(VERSION 3.18)

project(gtpin_offline_correlation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(offline_correlation
    offline_correlation.cpp
    offline_correlation_main.cpp
)

target_include_directories(offline_correlation PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(offline_correlation PRIVATE /W4)
else()
    target_compile_options(offline_correlation PRIVATE -Wall -Wextra -Wpedantic)
endif()
