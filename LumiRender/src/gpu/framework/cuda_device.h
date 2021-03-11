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
        };

        inline std::shared_ptr<Device> create_cuda_device() {
            return std::make_shared<Device>(std::make_unique<CUDADevice>());
        }
    }
}