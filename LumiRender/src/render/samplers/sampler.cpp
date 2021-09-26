//
// Created by Zero on 2021/2/25.
//


#include "sampler.h"
#include "render/include/creator.h"
#include "core/refl/factory.h"

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

        float Sampler::next_1d() {
            LUMINOUS_VAR_DISPATCH(next_1d)
        }

        float2 Sampler::next_2d() {
            LUMINOUS_VAR_DISPATCH(next_2d)
        }

        CPU_ONLY(Sampler Sampler::create(const SamplerConfig &config) {
            return detail::create<Sampler>(config);
        })

        REGISTER(Sampler)
    }
}