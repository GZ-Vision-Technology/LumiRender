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
            uint3 _grid_size = make_uint3(0);
            uint3 _block_size = make_uint3(0);
        public:
            CUDAKernel(CUfunction func) : _func(func) {}

            void configure(uint3 grid_size, uint3 local_size) override {

            }

            void launch(Dispatcher &dispatcher,
                        std::vector<void *> args) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                CU_CHECK(cuLaunchKernel(_func, _grid_size.x, _grid_size.y, _grid_size.z,
                                        _block_size.x, _block_size.y,_block_size.z,
                                        1024, stream, args.data(), nullptr));
            }
        };

        inline SP<Kernel> create_cuda_kernel(CUfunction func) {
            return std::make_shared<Kernel>(std::make_unique<CUDAKernel>(func));
        }
    }
}