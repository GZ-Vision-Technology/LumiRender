//
// Created by Zero on 2021/2/12.
//


#pragma once

#include "cuda.h"

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif

#ifdef _WIN32
#pragma warning( push )
#endif

#define OPTIX_CHECK(EXPR)                                                                                              \
    [&] {                                                                                                              \
        OptixResult res = EXPR;                                                                                        \
        if (res != OPTIX_SUCCESS) {                                                                                    \
            spdlog::error("OptiX call " #EXPR " failed with code {}: \"{}\" at {}:{}", int(res),                       \
                          optixGetErrorString(res), __FILE__, __LINE__);                                               \
            std::abort();                                                                                              \
        }                                                                                                              \
    }()

#define OPTIX_CHECK_WITH_LOG(EXPR, LOG)                                                                                \
    [&]{                                                                                                               \
        OptixResult res = EXPR;                                                                                        \
        if (res != OPTIX_SUCCESS)                                                                                      \
            spdlog::error("OptiX call " #EXPR " failed with code {}: \"{}\"\nLogs: {}", int(res),                      \
                      optixGetErrorString(res), LOG);                                                                  \
    } ()