cmake_minimum_required(VERSION 3.22.1)
project(brazier_sdcpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(module ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory(${module}/vendor/stable-diffusion.cpp sdcpp EXCLUDE_FROM_ALL)

add_library(brazier_sdcpp SHARED jni.cpp ${module}/cpp/brazier_sd.cpp)
target_include_directories(brazier_sdcpp PRIVATE ${module}/cpp)
target_link_libraries(brazier_sdcpp PRIVATE stable-diffusion log)
target_link_options(brazier_sdcpp PRIVATE -Wl,-z,max-page-size=16384)
