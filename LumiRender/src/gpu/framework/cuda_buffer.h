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
            CUdeviceptr _ptr;
            size_t _size_in_bytes;

        public:
            void *ptr() override { return (void *)_ptr; }

            CUDABuffer(size_t bytes) : _size_in_bytes(bytes) {
                CU_CHECK(cuMemAlloc(&_ptr, bytes));
            }

            ~CUDABuffer() { CU_CHECK(cuMemFree(_ptr)); }

            size_t size() const override { return _size_in_bytes; }

            void *address(size_t offset = 0) const override { return (void *)(_ptr + offset); }

            void download_async(Dispatcher &dispatcher, void *host_ptr, size_t size = 0, size_t offset = 0) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CU_CHECK(cuMemcpyDtoHAsync(host_ptr, _ptr + offset, size, stream));
            }

            void upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size = 0, size_t offset = 0) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CU_CHECK(cuMemcpyHtoDAsync(_ptr + offset, host_ptr, size, stream));
            }

            void download(void *host_ptr, size_t size = 0, size_t offset = 0) override {
                CU_CHECK(cuMemcpyDtoH(host_ptr, _ptr + offset, size));
            }

            void upload(const void *host_ptr, size_t size = 0, size_t offset = 0) override {
                CU_CHECK(cuMemcpyHtoD(_ptr + offset, host_ptr, size));
            }
        };
    }
}