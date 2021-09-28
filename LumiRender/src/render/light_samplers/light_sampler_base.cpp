//
// Created by Zero on 26/09/2021.
//

#include "light_sampler_base.h"
#include "render/lights/light.h"

namespace luminous {
    inline namespace render {

        CPU_ONLY(std::string SampledLight::to_string() const {
            return string_printf("sampled light :{PMF:%s, light:%s}", PMF, light->to_string().c_str());
        })

        const Light &LightSamplerBase::light_at(uint idx) const {
            return _lights[idx];
        }

        const Light &LightSamplerBase::infinite_light_at(index_t idx) const {
            return _infinite_lights[idx];
        }

    }
}