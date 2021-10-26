//
// Created by Zero on 2021/2/12.
//


#pragma once

#ifdef CUDA_SUPPORT

#include "base_libs/header.h"
#include "vector_types.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include "spdlog/spdlog.h"

#define CUDA_CHECK(EXPR)                                                                                               \
    [&] {                                                                                                              \
        if ((EXPR) != cudaSuccess) {                                                                                   \
            cudaError_t error = cudaGetLastError();                                                                    \
            spdlog::error("CUDA runtime error: {} at {}:{}", cudaGetErrorString(error), __FILE__, __LINE__);           \
            std::abort();                                                                                              \
        }                                                                                                              \
    }()

#define CU_CHECK(EXPR)                                                                                                 \
    [&] {                                                                                                              \
        CUresult result = EXPR;                                                                                        \
        if (result != CUDA_SUCCESS) {                                                                                  \
            const char *str;                                                                                           \
            assert(CUDA_SUCCESS == cuGetErrorString(result, &str));                                                    \
            spdlog::error("CUDA driver error: {} at {}:{}", str, __FILE__, __LINE__);                                  \
            std::abort();                                                                                              \
        }                                                                                                              \
    }()

namespace luminous {
    inline namespace gpu {

        template<typename T>
        void download(T * host_ptr, CUdeviceptr device_ptr, size_t num = 1, size_t offset = 0) {
            CU_CHECK(cuMemcpyDtoH(host_ptr, device_ptr + offset * sizeof(T), num * sizeof(T)));
        }

        /**
         * for debug, watch GPU memory content
         * @tparam T
         * @param ptr
         * @param num
         * @param offset
         * @return
         */
        template<typename T>
        std::vector<T> get_cuda_data(ptr_t ptr, size_t num = 1, size_t offset = 0) {
            std::vector<T> ret;
            ret.resize(num);
            download(ret.data(), ptr, num, offset);
            return ret;
        }

        template<typename T>
        T download(CUdeviceptr device_ptr, size_t offset = 0) {
            T ret;
            download<T>(&ret, device_ptr, 1, offset);
            return ret;
        }
    }
}

#endif