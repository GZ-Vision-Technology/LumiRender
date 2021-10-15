//
// Created by Zero on 26/08/2021.
//


#pragma once

#include "core/backend/device.h"
#include "core/backend/kernel.h"


namespace luminous {

    inline namespace utility {
        class ParallelForWorkPool;
    }

    inline namespace cpu {

        class CPUBuffer : public RawBuffer::Impl {
        private:
            void *_ptr{};
            const bool _is_external_ptr;
            size_t _size_in_bytes{0};
        public:
            LM_NODISCARD void *ptr() const override;

            explicit CPUBuffer(size_t bytes, void *ptr = nullptr);

            ~CPUBuffer() override;

            LM_NODISCARD size_t size() const override;

            LM_NODISCARD void *address(size_t offset) const override;

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

            LM_NODISCARD uint64_t tex_handle() const override;

            void copy_from(Dispatcher &dispatcher, const Image &image) override;

            LM_NODISCARD Image download() const override;

            void copy_from(const Image &image) override;

            ~CPUTexture() override;
        };

        class CPUKernel : public Kernel::Impl {
        public:
            using func_type = std::function<void(uint, void *[])>;
        private:
            func_type _func;
        public:
            explicit CPUKernel(func_type func);

            void configure(uint3 grid_size, uint3 local_size, size_t sm) override {}

            void launch(Dispatcher &dispatcher, void *args[]) override;
        };

        inline std::shared_ptr<Kernel> create_cpu_kernel(const CPUKernel::func_type &func) {
            return std::make_shared<Kernel>(std::make_unique<CPUKernel>(func));
        }

        class CPUDispatcher : public Dispatcher::Impl {
        private:
            ParallelForWorkPool *_work_pool{};
        public:
            CPUDispatcher();

            void wait() override;

            void then(std::function<void(void)> F) override;

            ~CPUDispatcher() override;
        };

        class CPUDevice : public Device::Impl {
        public:
            CPUDevice() = default;

            RawBuffer create_buffer(size_t bytes, void *ptr) override;

            LM_NODISCARD bool is_cpu() const override { return true; }

            DTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) override;

            Dispatcher new_dispatcher() override;

            ~CPUDevice() override = default;
        };

        inline std::unique_ptr<Device> create_cpu_device() {
            return std::make_unique<Device>(std::make_unique<CPUDevice>());
        }
    }
}