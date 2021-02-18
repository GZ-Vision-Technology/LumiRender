//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "core/context.h"
#include "cuda_dispatcher.h"
#include "backend/device.h"
#include "cuda_buffer.h"

namespace luminous {
    inline namespace gpu {

        class CUDADevice : public Device::Impl {
        public:
            RawBuffer allocate_buffer(size_t bytes) override {
                void *ptr;
                CUDA_CHECK(cudaMalloc(&ptr, bytes));
                return RawBuffer(std::make_unique<CUDABuffer>(ptr, bytes));
            }

            Dispatcher new_dispatcher() override {
                cudaStream_t stream;
                CUDA_CHECK(cudaStreamCreate(&stream));
                return Dispatcher(std::make_unique<CUDADispatcher>(stream));
            }
        };
    }
}