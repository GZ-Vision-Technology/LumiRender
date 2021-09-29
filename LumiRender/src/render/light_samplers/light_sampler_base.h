//
// Created by Zero on 2021/4/9.
//


#pragma once

#include "base_libs/math/common.h"
#include "render/lights/light_util.h"
#include "core/backend/buffer_view.h"
#include "core/refl/factory.h"

namespace luminous {
    inline namespace render {

        class Light;

        class Envmap;

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

            CPU_ONLY(LM_NODISCARD std::string to_string() const;)
        };

        class LightSamplerBase : BASE_CLASS() {
        public:
            REFL_CLASS(LightSamplerBase)

        protected:
            DEFINE_AND_REGISTER_MEMBER(BufferView<const Light>, _lights)

            DEFINE_AND_REGISTER_MEMBER(BufferView<const Light>, _infinite_lights)

        public:
            NDSC_XPU_INLINE BufferView<const Light> lights() const {
                return _lights;
            }

            NDSC_XPU_INLINE BufferView<const Light> infinite_lights() const {
                return _infinite_lights;
            }

            NDSC_XPU const Light &light_at(uint idx) const;

            NDSC_XPU const Light &infinite_light_at(index_t idx) const;

            template<typename Func>
            XPU void for_each_light(const Func &func) const {
                for (int i = 0; i < light_num(); ++i) {
                    func(light_at(i), i);
                }
            }

            template<typename Func>
            XPU void for_each_infinite_light(const Func &func) const {
                for (int i = 0; i < infinite_light_num(); ++i) {
                    func(infinite_light_at(i), i);
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