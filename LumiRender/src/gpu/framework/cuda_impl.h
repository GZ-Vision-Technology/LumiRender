//
// Created by Zero on 2021/4/14.
//


#pragma once

#include "helper/optix.h"
#include "core/backend/buffer.h"
#include "core/backend/texture.h"
#include "core/backend/dispatcher.h"
#include "core/backend/device.h"
#include "core/backend/module.h"
#include <cuda.h>

namespace luminous {
    inline namespace gpu {
        class CUDATexture : public DTexture::Impl {
        private:
            CUtexObject _tex_handle{};
            CUarray _array_handle{};
            CUsurfObject _surface_handle{};

            LM_NODISCARD CUDA_MEMCPY2D common_memcpy_from_desc() const;

            LM_NODISCARD CUDA_MEMCPY2D host_src_memcpy_desc(const Image &image) const;

        public:
            CUDATexture(PixelFormat pixel_format, uint2 resolution);

            void init();

            LM_NODISCARD uint64_t tex_handle() const override;

            void copy_from(Dispatcher &dispatcher, const Image &image) override;

            LM_NODISCARD Image download() const override;

            void copy_from(const Image &image) override;

            ~CUDATexture() override;
        };

        class CUDADispatcher : public Dispatcher::Impl {
        public:
            CUstream stream{};

            CUDADispatcher();

            void wait() override;

            void then(std::function<void(void)> F) override;

            ~CUDADispatcher() override;
        };

        class CUDABuffer : public RawBuffer::Impl {
        private:
            CUdeviceptr _ptr{};
            size_t _size_in_bytes;
            const bool _is_external_ptr;
        public:
            LM_NODISCARD void *ptr() const override;

            explicit CUDABuffer(size_t bytes, void *ptr = nullptr);

            ~CUDABuffer() override;

            LM_NODISCARD size_t size() const override;

            LM_NODISCARD void *address(size_t offset) const override;

            void memset(uint32_t val) override;

            void download_async(Dispatcher &dispatcher, void *host_ptr, size_t size, size_t offset) override;

            void upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size, size_t offset) override;

            void download(void *host_ptr, size_t size, size_t offset) override;

            void upload(const void *host_ptr, size_t size, size_t offset) override;
        };

        class CUDADevice : public Device::Impl {
        private:
            CUdevice _cu_device{};
            CUcontext _cu_context{};

        public:
            CUDADevice();

            RawBuffer create_buffer(size_t bytes, void *ptr) override;

            DTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) override;

            LM_NODISCARD bool is_cpu() const override { return false; }

            Dispatcher new_dispatcher() override;

            ~CUDADevice() override;
        };

        inline std::unique_ptr<Device> create_cuda_device() {
            return std::make_unique<Device>(std::make_unique<CUDADevice>());
        }

        class CUDAModule : public Module::Impl {
        private:
            CUmodule _module{};
        public:
            explicit CUDAModule(const std::string &ptx_code);

            LM_NODISCARD uint64_t get_kernel_handle(const std::string &name) override;

            LM_NODISCARD std::pair<ptr_t, size_t> get_global_var(const std::string &name) override;

            virtual void upload_data_to_global_var(const std::string &name, const void *data, size_t size) override;
        };

        inline SP<Module> create_cuda_module(const std::string &ptx_code) {
            return std::make_shared<Module>(std::make_unique<CUDAModule>(ptx_code));
        }
    }
}