//
// Created by Zero on 2021/4/14.
//


#pragma once

#include "helper/optix.h"
#include "core/backend/buffer.h"
#include "core/backend/texture.h"
#include "core/backend/dispatcher.h"
#include "core/backend/kernel.h"
#include "core/backend/device.h"
#include "core/backend/module.h"

namespace luminous {
    inline namespace gpu {
        class CUDATexture : DeviceTexture::Impl {
        private:
            CUtexObject _tex_handle;
            CUarray _array_handle;
            CUsurfObject _surf_handle;
        public:
            CUDATexture(PixelFormat pixel_format, uint2 resolution);

            void init();

            void copy_to(Dispatcher &dispatcher, const Image &image) const;

            void copy_to(Dispatcher &dispatcher, Buffer<> &buffer) const;

            void copy_from(Dispatcher &dispatcher, const Buffer<> &buffer);

            void copy_from(Dispatcher &dispatcher, const Image &image);

            ~CUDATexture();
        };

        class CUDADispatcher : public Dispatcher::Impl {
        public:
            CUstream stream;

            CUDADispatcher();

            void wait() override;

            void then(std::function<void(void)> F) override;

            ~CUDADispatcher();
        };

        class CUDABuffer : public RawBuffer::Impl {
        private:
            CUdeviceptr _ptr{};
            size_t _size_in_bytes;

        public:
            void *ptr() override;

            CUDABuffer(size_t bytes);

            ~CUDABuffer();

            size_t size() const override;

            void *address(size_t offset = 0) const override;

            void download_async(Dispatcher &dispatcher, void *host_ptr, size_t size, size_t offset) override;

            void upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size, size_t offset) override;

            void download(void *host_ptr, size_t size, size_t offset) override;

            void upload(const void *host_ptr, size_t size, size_t offset) override;
        };

        class CUDAKernel : public Kernel::Impl {
        private:
            CUfunction _func{};
            uint3 _grid_size = make_uint3(1);
            uint3 _block_size = make_uint3(5);
            int _auto_block_size = 0;
            int _min_grid_size = 0;
            size_t _shared_mem = 1024;
        public:
            CUDAKernel(CUfunction func);

            void compute_fit_size();

            void configure(uint3 grid_size, uint3 local_size, size_t sm = 0) override;

            void launch(Dispatcher &dispatcher, int n_items,
                        std::vector<void *> &args) override;

            void launch(Dispatcher &dispatcher, std::vector<void *> &args) override;
        };

        inline std::shared_ptr<Kernel> create_cuda_kernel(CUfunction func) {
            return std::make_shared<Kernel>(std::make_unique<CUDAKernel>(func));
        }

        class CUDADevice : public Device::Impl {
        private:
            CUdevice  _cu_device{};
            CUcontext _cu_context{};
        public:
            CUDADevice();

            RawBuffer allocate_buffer(size_t bytes) override;

            DeviceTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) override;

            Dispatcher new_dispatcher() override;

            ~CUDADevice();
        };

        inline std::shared_ptr<Device> create_cuda_device() {
            return std::make_shared<Device>(std::make_unique<CUDADevice>());
        }

        class CUDAModule : public Module::Impl {
        private:
            CUmodule _module{};
        public:
            explicit CUDAModule(const std::string &ptx_code);

            SP<Kernel> get_kernel(const std::string &name);
        };

        inline SP<Module> create_cuda_module(const std::string &ptx_code) {
            return std::make_shared<Module>(std::make_unique<CUDAModule>(ptx_code));
        }
    }
}