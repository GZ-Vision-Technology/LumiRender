//
// Created by Zero on 2021/2/12.
//


#pragma once

#include "helper/optix.h"
#include "core/backend/buffer.h"
#include "cuda_dispatcher.h"

namespace luminous {
    inline namespace gpu {
        class CUDABuffer : public RawBuffer::Impl {
        private:
            void *_ptr;
            size_t _bytes;

        public:
            void *device_ptr() { return _ptr; }

            void *ptr() override { return _ptr; }

            CUDABuffer(size_t bytes) : _bytes(bytes) {
                CUDA_CHECK(cudaMalloc(&_ptr, bytes));
            }

            ~CUDABuffer() { CUDA_CHECK(cudaFree(_ptr)); }

            size_t size() const override { return _bytes; }

            void download_async(Dispatcher &dispatcher, void *host_data, size_t size, size_t offset = 0) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CUDA_CHECK(cudaMemcpyAsync(host_data, (const uint8_t *) _ptr + offset, size, cudaMemcpyDeviceToHost,
                                           stream));
            }

            void upload_async(Dispatcher &dispatcher, const void *host_data, size_t size, size_t offset = 0) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) _ptr + offset, host_data, size, cudaMemcpyHostToDevice, stream));
            }

            void download(void *host_data, size_t size, size_t offset = 0) override {
                CUDA_CHECK(cudaMemcpy(host_data, (const uint8_t *) _ptr + offset, size, cudaMemcpyDeviceToHost));
            }

            void upload(const void *host_data, size_t size, size_t offset = 0) override {
                CUDA_CHECK(cudaMemcpy((uint8_t *) _ptr + offset, host_data, size, cudaMemcpyHostToDevice));
            }
        };
    }
}