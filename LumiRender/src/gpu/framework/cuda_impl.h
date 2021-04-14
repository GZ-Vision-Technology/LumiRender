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
        class CUDATexture : Texture::Impl {

        };

        class CUDADispatcher : public Dispatcher::Impl {
        public:
            CUstream stream;

            CUDADispatcher() {
                CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
            }

            void wait() override {CU_CHECK(cuStreamSynchronize(stream)); }

            void then(std::function<void(void)> F) override {
                using Func = std::function<void(void)>;
                Func *f = new Func(std::move(F));
                auto wrapper = [](void *p) {
                    auto f = reinterpret_cast<Func *>(p);
                    (*f)();
                    delete f;
                };
                CU_CHECK(cuLaunchHostFunc(stream, wrapper, (void *) f));
            }

            ~CUDADispatcher() {CU_CHECK(cuStreamDestroy(stream)); }
        };

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



        class CUDAKernel : public Kernel::Impl {
        private:
            CUfunction _func{};
            uint3 _grid_size = make_uint3(1);
            uint3 _block_size = make_uint3(5);
            int _auto_block_size = 0;
            int _min_grid_size = 0;
            size_t _shared_mem = 1024;
        public:
            CUDAKernel(CUfunction func) : _func(func) {
                compute_fit_size();
            }

            void compute_fit_size() {
                cuOccupancyMaxPotentialBlockSize(&_min_grid_size, &_auto_block_size, _func, 0, _shared_mem, 0);
            }

            void configure(uint3 grid_size, uint3 local_size, size_t sm = 0) override {
                _shared_mem = sm == 0 ? _shared_mem : sm;
                _grid_size = grid_size;
                _block_size = local_size;
            }

            void launch(Dispatcher &dispatcher, int n_items,
                        std::vector<void *> &args) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                int grid_size = (n_items + _auto_block_size - 1) / _auto_block_size;
                CU_CHECK(cuLaunchKernel(_func, grid_size, 1, 1,
                                        _auto_block_size, 1, 1,
                                        _shared_mem, stream, args.data(), nullptr));
            }

            void launch(Dispatcher &dispatcher, std::vector<void *> &args) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CU_CHECK(cuLaunchKernel(_func, _grid_size.x, _grid_size.y, _grid_size.z,
                                        _block_size.x, _block_size.y,_block_size.z,
                                        _shared_mem, stream, args.data(), nullptr));
            }
        };

        inline std::shared_ptr<Kernel> create_cuda_kernel(CUfunction func) {
            return std::make_shared<Kernel>(std::make_unique<CUDAKernel>(func));
        }

        class CUDADevice : public Device::Impl {
        private:
            CUdevice  _cu_device{};
            CUcontext _cu_context{};
        public:
            CUDADevice() {
                CU_CHECK(cuInit(0));
                CU_CHECK(cuDeviceGet(&_cu_device, 0));
                CU_CHECK(cuCtxCreate(&_cu_context, 0, _cu_device));
                CU_CHECK(cuCtxSetCurrent(_cu_context));
            }

            RawBuffer allocate_buffer(size_t bytes) override {
                return RawBuffer(std::make_unique<CUDABuffer>(bytes));
            }

            Dispatcher new_dispatcher() override {
                return Dispatcher(std::make_unique<CUDADispatcher>());
            }

            ~CUDADevice() {
                CU_CHECK(cuCtxDestroy(_cu_context));
            }
        };

        inline std::shared_ptr<Device> create_cuda_device() {
            return std::make_shared<Device>(std::make_unique<CUDADevice>());
        }

        class CUDAModule : public Module::Impl {
        private:
            CUmodule _module;
        public:
            explicit CUDAModule(const std::string &ptx_code) {
                CU_CHECK(cuModuleLoadData(&_module, ptx_code.c_str()));
            }

            SP<Kernel> get_kernel(const std::string &name) {
                CUfunction func;
                CU_CHECK(cuModuleGetFunction(&func, _module, name.c_str()));
                return create_cuda_kernel(func);
            }
        };

        inline SP<Module> create_cuda_module(const std::string &ptx_code) {
            return std::make_shared<Module>(std::make_unique<CUDAModule>(ptx_code));
        }
    }
}