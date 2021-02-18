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
            size_t bytes;

        public:
            void *device_ptr() { return _ptr; }

            void *ptr() override { return _ptr; }

            CUDABuffer(void *ptr, size_t bytes) : _ptr(ptr), bytes(bytes) {}

            ~CUDABuffer() {CUDA_CHECK(cudaFree(_ptr)); }

            size_t size() const override { return bytes; }

            void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_data) {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CUDA_CHECK(cudaMemcpyAsync(host_data, (const uint8_t *) _ptr + offset, size, cudaMemcpyDeviceToHost,
                                           stream));
            }

            void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) _ptr + offset, host_data, size, cudaMemcpyHostToDevice, stream));
            }
        };
    }
}