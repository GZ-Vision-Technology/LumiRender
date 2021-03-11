//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/kernel.h"
#include "gpu/framework/helper/cuda.h"

namespace luminous {
    inline namespace gpu {

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

        inline SP<Kernel> create_cuda_kernel(CUfunction func) {
            return std::make_shared<Kernel>(std::make_unique<CUDAKernel>(func));
        }
    }
}