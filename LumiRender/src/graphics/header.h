//
// Created by Zero on 2021/2/4.
//


#pragma once



#include <assert.h>

#if defined(__CUDACC__)
    #define XPU __host__ __device__
    #define GPU __device__
    #define CPU __host__
#else
    #define XPU
    #define GPU
    #define CPU
#endif

#define XPU_INLINE XPU inline

#define NDSC [[nodiscard]]

#if defined(__CUDA_ARCH__)
    #define IS_GPU_CODE
#endif

#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

#define USE_LSTD 1

#define DCHECK(a) assert(a);
#define DCHECK_EQ(a, b) DCHECK(a == b)
#define DCHECK_GT(a, b) DCHECK(a > b);
#define DCHECK_GE(a, b) DCHECK(a >= b);
#define DCHECK_LT(a, b) DCHECK(a < b);
#define DCHECK_LE(a, b) DCHECK(a <= b);

