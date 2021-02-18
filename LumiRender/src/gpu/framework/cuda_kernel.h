//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "backend/kernel.h"


namespace luminous {
    inline namespace gpu {
        class CUDAKernel : public Kernel::Impl {
        private:
            CUfunction func;

        public:
            CUDAKernel(CUfunction func) : func(func) {}

            void launch(Dispatcher &dispatcher,
                        std::vector<void *> args,
                        uint3 global_size,
                        uint3 local_size) override {
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
                vec3 grid_size = (global_size + local_size - uvec3(1)) / local_size;
                CU_CHECK(cuLaunchKernel(func, grid_size.x, grid_size.y, grid_size.z, local_size.x, local_size.y,
                                        local_size.z, 1024, stream, args.data(), nullptr));
            }
        };
    }
}