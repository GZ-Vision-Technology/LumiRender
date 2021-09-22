//
// Created by Zero on 2021/4/9.
//


#pragma once

#include "base_libs/math/common.h"
#include "render/lights/light.h"
#include "core/backend/buffer_view.h"
#include "core/concepts.h"

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
            BufferView<const Light> _infinite_lights;
        public:
            NDSC_XPU_INLINE BufferView<const Light> lights() const {
                return _lights;
            }

            NDSC_XPU_INLINE BufferView<const Light> infinite_lights() const {
                return _infinite_lights;
            }

            NDSC_XPU_INLINE const Light &light_at(uint idx) const {
                return _lights[idx];
            }

            NDSC_XPU_INLINE const Light &infinite_light_at(index_t idx) const {
                return _infinite_lights[idx];
            }

            template<typename Func>
            XPU_INLINE void for_each_light(Func func) const {
                for (int i = 0; i < light_num(); ++i) {
                    func(light_at(i), i);
                }
            }

            template<typename Func>
            XPU_INLINE void for_each_infinite_light(Func func) const {
                for (int i = 0; i < infinite_light_num(); ++i) {
                    func(*infinite_light_at(i).template get<Envmap>(), i);
                }
            }

            XPU_INLINE void set_lights(BufferView<const Light> lights) {
                _lights = lights;
            }

            XPU_INLINE void set_infinite_lights(BufferView<const Light> lights) {
                _infinite_lights = lights;
            }

            NDSC_XPU size_t light_num() const {
                return _lights.size();
            }

            NDSC_XPU_INLINE size_t infinite_light_num() const {
                return _infinite_lights.size();
            }
        };
    }
}