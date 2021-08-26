//
// Created by Zero on 26/08/2021.
//


#pragma once

#include "core/backend/device.h"

namespace luminous {
    inline namespace cpu {

        class CPUBuffer : public RawBuffer::Impl {
        private:
            void *_ptr{};
            size_t _size_in_bytes{0};
        public:
            void *ptr() override;

            explicit CPUBuffer(size_t bytes);

            ~CPUBuffer() override;

            NDSC size_t size() const override;

            NDSC void *address(size_t offset) const override;

            void memset(uint32_t val) override;

            void download_async(Dispatcher &dispatcher, void *host_ptr, size_t size, size_t offset) override;

            void upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size, size_t offset) override;

            void download(void *host_ptr, size_t size, size_t offset) override;

            void upload(const void *host_ptr, size_t size, size_t offset) override;
        };

//        class CPUDevice : public Device::Impl {
//        public:
//            CPUDevice() = default;
//
//            RawBuffer allocate_buffer(size_t bytes) override;
//
//            DTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) override;
//
//            Dispatcher new_dispatcher() override;
//
//            ~CPUDevice() override = default;
//        };
    }
}