//
// Created by Zero on 16/02/2022.
//


#pragma once

#include "base_libs/common.h"
#include <OpenImageDenoise/oidn.hpp>
#include <iostream>

namespace luminous {
    inline namespace denoise {
        class Denoiser {
        private:
            oidn::DeviceRef _device{};
        public:
            Denoiser() : _device(oidn::newDevice()) {
                _device.commit();
            }

            void execute(uint2 res, float4 *output, float4 *color,
                         float4 *normal = nullptr, float4 *albedo = nullptr) {
                oidn::FilterRef filter = _device.newFilter("RT");
                filter.setImage("output", output, oidn::Format::Float3, res.x, res.y, 0, sizeof(float4));
                filter.setImage("color", color, oidn::Format::Float3, res.x, res.y, 0, sizeof(float4));
                if (normal) {
                    filter.setImage("normal", normal, oidn::Format::Float3, res.x, res.y, 0, sizeof(float4));
                }
                if (albedo) {
                    filter.setImage("albedo", albedo, oidn::Format::Float3, res.x, res.y, 0, sizeof(float4));
                }
                // color image is HDR
                filter.set("hdr", true);
                filter.commit();
                filter.execute();

                const char *errorMessage;
                if (_device.getError(errorMessage) != oidn::Error::None) {
                    std::cout << "oidn Error: " << errorMessage << std::endl;
                }
            }
        };
    }
}