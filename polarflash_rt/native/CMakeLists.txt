cmake_minimum_required(VERSION 3.20)
project(polarflash_rt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(polarflash_rt_native SHARED
    polarflash_simd.cpp
    polarflash_simd.hpp
)

target_include_directories(polarflash_rt_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(polarflash_rt_native PRIVATE /arch:AVX2 /O2)
else()
    target_compile_options(polarflash_rt_native PRIVATE -mavx2 -mfma -O3 -fopenmp)
endif()

