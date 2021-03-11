//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/kernel.h"
#include "gpu/framework/helper/cuda.h"

namespace luminous {
    inline namespace gpu {

        class CUDAKernel : public Kernel::Impl {
        public:
            CUfunction _func;
            uint3 _grid_size = make_uint3(1);
            uint3 _block_size = make_uint3(5);
            size_t _shared_mem = 1024;
        public:
            CUDAKernel(CUfunction func) : _func(func) {}

            void configure(uint3 grid_size, uint3 local_size, size_t sm = 0) override {
                _shared_mem = sm == 0 ? _shared_mem : sm;
            }

            void launch(Dispatcher &dispatcher,
                        std::vector<void *> args) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CU_CHECK(cuLaunchKernel(_func, _grid_size.x, _grid_size.y, _grid_size.z,
                                        _block_size.x, _block_size.y,_block_size.z,
                                        _shared_mem, stream, args.data(), nullptr));
            }
        };

        inline SP<Kernel> create_cuda_kernel(CUfunction func) {
            return std::make_shared<Kernel>(std::make_unique<CUDAKernel>(func));
        }
    }
}