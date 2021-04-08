//
// Created by Zero on 2021/4/7.
//

#include "light_handle.h"

namespace luminous {
    inline namespace render {

        LightType LightHandle::type() const {
            LUMINOUS_VAR_DISPATCH(type);
        }

        const char *LightHandle::name() {
            LUMINOUS_VAR_DISPATCH(name);
        }

        Interaction LightHandle::sample(float u,const HitGroupData * hit_group_data) const {
            LUMINOUS_VAR_DISPATCH(sample, u,hit_group_data);
        }

        LightLiSample LightHandle::Li(LightLiSample lls) const {
            LUMINOUS_VAR_DISPATCH(Li, lls);
        }

        bool LightHandle::is_delta() const {
            LUMINOUS_VAR_DISPATCH(is_delta);
        }

        float LightHandle::PDF_Li(const Interaction &ref_p, const Interaction &p_light) const {
            LUMINOUS_VAR_DISPATCH(PDF_Li, ref_p, p_light);
        }

        float3 LightHandle::power() const {
            LUMINOUS_VAR_DISPATCH(power);
        }

        std::string LightHandle::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string);
        }

        namespace detail {
            template<uint8_t current_index>
            NDSC LightHandle create_light(const LightConfig &config) {
                using Light = std::remove_pointer_t<std::tuple_element_t<current_index, LightHandle::TypeTuple>>;
                if (Light::name() == config.type) {
                    return LightHandle(Light::create(config));
                }
                return create_light<current_index + 1>(config);
            }

            template<>
            NDSC LightHandle create_light<std::tuple_size_v<LightHandle::TypeTuple>>(const LightConfig &config) {
                LUMINOUS_ERROR("unknown sampler type:", config.type);
            }
        }

        LightHandle LightHandle::create(const LightConfig &config) {
            return detail::create_light<0>(config);
        }
    } // luminous::render
} // luminous