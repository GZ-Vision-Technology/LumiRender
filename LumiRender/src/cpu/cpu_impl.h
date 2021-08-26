//
// Created by Zero on 26/08/2021.
//


#pragma once

#include "core/backend/device.h"

namespace luminous {
    inline namespace cpu {
        class CPUDevice : public Device::Impl {
        public:
            CPUDevice() = default;

            RawBuffer allocate_buffer(size_t bytes) override {

            }

            DeviceTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) override {

            }

            Dispatcher new_dispatcher() override {

            }

            ~CPUDevice() override = default;
        };
    }
}