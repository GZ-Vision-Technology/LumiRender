//
// Created by Zero on 2021/2/25.
//


#include "sampler.h"

namespace luminous {

    inline namespace render {
        int Sampler::spp() const {
            LUMINOUS_VAR_DISPATCH(spp)
        }

        SensorSample Sampler::sensor_sample(uint2 p_raster) {
            SensorSample ss;
            ss.p_film = make_float2(p_raster) + next_2d();
            ss.p_lens = next_2d();
            ss.time = next_1d();
            return ss;
        }

        void Sampler::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            LUMINOUS_VAR_DISPATCH(start_pixel_sample, pixel, sample_index, dimension)
        }

        const char *Sampler::name() {
            LUMINOUS_VAR_DISPATCH(name)
        }

        float Sampler::next_1d() {
            LUMINOUS_VAR_DISPATCH(next_1d)
        }

        float2 Sampler::next_2d() {
            LUMINOUS_VAR_DISPATCH(next_2d)
        }

        std::string Sampler::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string)
        }

        namespace detail {
            template<uint8_t current_index>
            NDSC Sampler create_sampler(const SamplerConfig &config) {
                using Class = std::remove_pointer_t<std::tuple_element_t<current_index, Sampler::TypeTuple>>;
                if (Class::name() == config.type) {
                    return Sampler(Class::create(config));
                }
                return create_sampler<current_index + 1>(config);
            }

            template<>
            NDSC Sampler
            create_sampler<std::tuple_size_v<Sampler::TypeTuple>>(const SamplerConfig &config) {
                LUMINOUS_ERROR("unknown sampler type:", config.type);
            }
        }

        Sampler Sampler::create(const SamplerConfig &config) {
            return detail::create_sampler<0>(config);
        }
    }
}