//
// Created by Zero on 2021/2/12.
//


#pragma once

#include "graphics/header.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call)                                              \
{                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
        fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
        throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
}

#define CUDA_CHECK2( where, call )                                      \
{                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
}

#define CUDA_SYNC_CHECK()                                       \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      throw std::runtime_error("fatal cuda error");             \
    }                                                           \
  }


#define CUDA_CHECK_NOTHROW( call )                                      \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }


#define CUDA_CHECK2_NOTHROW( where, call )                              \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }
