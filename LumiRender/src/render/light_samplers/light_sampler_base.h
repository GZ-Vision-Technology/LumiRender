//
// Created by Zero on 2021/4/9.
//


#pragma once

#include "graphics/math/common.h"
#include "render/lights/light.h"
#include "core/backend/buffer_view.h"

namespace luminous {
    inline namespace render {
        struct SampledLight {
            const Light *light{nullptr};
            float PMF{0.f};

            XPU SampledLight() = default;

            XPU SampledLight(const Light *light, float PMF)
                    : light(light), PMF(PMF) {
                DCHECK(PMF > 0)
            }

            NDSC_XPU_INLINE bool is_valid() const {
                return light != nullptr;
            }

            GEN_STRING_FUNC({
                return string_printf("sampled light :{PMF:%s, light:%s}",
                                     PMF, light->to_string().c_str());
            })
        };

        class LightSamplerBase {
        protected:
            BufferView<const Light> _lights;
        public:
            NDSC_XPU BufferView<const Light> lights() const {
                return _lights;
            }

            NDSC_XPU const Light &light_at(uint idx) const {
                return _lights[idx];
            }

            template<typename Func>
            XPU void for_each_light(Func func) const {
                for (int i = 0; i < light_num(); ++i) {
                    func(light_at(i), i);
                }
            }

            template<typename Func>
            XPU void for_each_infinity_light(Func func) const {
                for (int i = 0; i < light_num(); ++i) {
                    const Light &light = light_at(i);
                    if (light.is_infinity()) {
                        func(_lights[i], i);
                    }
                }
            }

            XPU void set_lights(BufferView<const Light> lights) {
                _lights = lights;
            }

            NDSC_XPU size_t light_num() const {
                return _lights.size();
            }
        };
    }
}