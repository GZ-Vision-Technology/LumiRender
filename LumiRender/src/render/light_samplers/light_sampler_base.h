//
// Created by Zero on 2021/4/9.
//


#pragma once

#include "graphics/math/common.h"
#include "render/lights/light_handle.h"

namespace luminous {
    inline namespace render {
        struct SampledLight {
            LightHandle light;
            float PMF = -1;

            bool valid() const {
                return PMF != -1;
            }

            NDSC std::string to_string() const {
                return string_printf("sampled light :{PMF:%s, light:%s}",
                                     PMF, light.to_string().c_str());
            }
        };

        class LightSamplerBase {
        protected:
            const LightHandle *_host_lights{nullptr};
            const LightHandle *_device_lights{nullptr};
            size_t _num_lights{0};
            const LightHandle * lights() const {
                return _device_lights;
            }
        };
    }
}