//
// Created by Zero on 2020/9/3.
//

#pragma once

#define XPU
#define GPU
#define CPU

#if defined(__CUDACC__)
    #define XPU __host__ __device__
    #define GPU __device__
    #define CPU __host__
#endif

#define F_INLINE __forceinline

#include "scalar_types.h"
#include "vector_types.h"
#include "matrix_types.h"

