//
// Created by Zero on 2020/8/31.
//

#pragma once

#include "macro.h"

#include "logging.h"

#if defined(_MSC_VER)
#define HAVE_ALIGNED_MALLOC
#endif

#ifndef L1_CACHE_LINE_SIZE
#define L1_CACHE_LINE_SIZE 64
#endif


#define F_INLINE __forceinline

#if defined(__CUDACC__)
    #define IS_GPU_CODE
#endif

#define HAVE_POSIX_MEMALIGN


