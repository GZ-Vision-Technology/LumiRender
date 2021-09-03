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
            NDSC void *ptr() const override;

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

        class CPUTexture : public DTexture::Impl {
        private:
            uint64_t _handle{};
        public:
            CPUTexture(PixelFormat pixel_format, uint2 resolution);

            void init();

            NDSC uint64_t tex_handle() const override;

            void copy_from(Dispatcher &dispatcher, const Image &image) override;

            NDSC Image download() const override;

            void copy_from(const Image &image) override;

            ~CPUTexture() override;
        };

        class CPUDispatcher : public Dispatcher::Impl {
        public:
            CPUDispatcher();

            void wait() override;

            void then(std::function<void(void)> F) override;

            ~CPUDispatcher() override;
        };

        class CPUDevice : public Device::Impl {
        public:
            CPUDevice() = default;

            RawBuffer allocate_buffer(size_t bytes) override;

            DTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) override;

            std::shared_ptr<Scene> create_scene(Device *device, Context *context) override;

            Dispatcher new_dispatcher() override;

            ~CPUDevice() override = default;
        };

        inline std::unique_ptr<Device> create_cpu_device() {
            return std::make_unique<Device>(std::make_unique<CPUDevice>());
        }
    }
}