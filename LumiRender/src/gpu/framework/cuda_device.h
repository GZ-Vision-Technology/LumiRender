//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "core/context.h"
#include "cuda_dispatcher.h"
#include "core/backend/device.h"
#include "cuda_buffer.h"

namespace luminous {
    inline namespace gpu {

        class CUDADevice : public Device::Impl {
        public:
            RawBuffer allocate_buffer(size_t bytes) override {
                return RawBuffer(std::make_unique<CUDABuffer>(bytes));
            }

            Dispatcher new_dispatcher() override {
                return Dispatcher(std::make_unique<CUDADispatcher>());
            }
        };
    }
}