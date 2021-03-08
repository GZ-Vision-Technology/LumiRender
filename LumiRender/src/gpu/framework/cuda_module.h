//
// Created by Zero on 2021/3/9.
//


#pragma once

#include "core/backend/module.h"
#include "core/context.h"
#include "cuda_kernel.h"

namespace luminous {
    inline namespace gpu {
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