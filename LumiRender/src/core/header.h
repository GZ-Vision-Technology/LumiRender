//
// Created by Zero on 2020/8/31.
//

#pragma once

#include "macro.h"
#include "ext/nlohmann/json.hpp"
#include "logging.h"

#if defined(_MSC_VER)
#define HAVE_ALIGNED_MALLOC
#endif

#ifndef L1_CACHE_LINE_SIZE
#define L1_CACHE_LINE_SIZE 64
#endif

#define XPU
#define GPU
#define CPU

#if defined(__CUDACC__)
    #define XPU __host__ __device__
    #define GPU __device__
    #define CPU __host__
#endif

#define F_INLINE __forceinline

#if defined(__CUDA_ARCH__)
    #define IS_GPU_CODE
#endif

#define HAVE_POSIX_MEMALIGN


using DataWrap = nlohmann::json ;
