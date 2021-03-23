//
// Created by Zero on 2021/2/25.
//


#include "sampler_handle.h"

namespace luminous {

    inline namespace render {
        int SamplerHandle::spp() const {
            LUMINOUS_VAR_DISPATCH(spp)
        }

        SensorSample SamplerHandle::sensor_sample(int2 p_raster) {
            SensorSample ss;
            ss.p_film = make_float2(p_raster) + next_2d();
            ss.p_lens = next_2d();
            ss.time = next_1d();
            return ss;
        }

        void SamplerHandle::start_pixel_sample(uint2 pixel, int sample_index, int dimension) {
            LUMINOUS_VAR_DISPATCH(start_pixel_sample, pixel, sample_index, dimension)
        }

        const char *SamplerHandle::name() {
            LUMINOUS_VAR_DISPATCH(name)
        }

        float SamplerHandle::next_1d() {
            LUMINOUS_VAR_DISPATCH(next_1d)
        }

        float2 SamplerHandle::next_2d() {
            LUMINOUS_VAR_DISPATCH(next_2d)
        }

        std::string SamplerHandle::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string)
        }

        namespace detail {
            template<uint8_t current_index>
            NDSC SamplerHandle create_sampler(const SamplerConfig &config) {
                using Sampler = std::remove_pointer_t<std::tuple_element_t<current_index, SamplerHandle::TypeTuple>>;
                if (Sampler::name() == config.type) {
                    return SamplerHandle(Sampler::create(config));
                }
                return create_sampler<current_index + 1>(config);
            }

            template<>
            NDSC SamplerHandle
            create_sampler<std::tuple_size_v<SamplerHandle::TypeTuple>>(const SamplerConfig &config) {
                LUMINOUS_ERROR("unknown sampler type:", config.type);
            }
        }

        SamplerHandle SamplerHandle::create(const SamplerConfig &config) {
            return detail::create_sampler<0>(config);
        }
    }
}