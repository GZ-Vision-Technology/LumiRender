//
// Created by Zero on 2021/2/25.
//


#include "sampler.h"
#include "render/filters/shader_include.h"
#include "render/include/creator.h"
#include "core/refl/factory.h"

namespace luminous {

    inline namespace render {
        int Sampler::spp() const {
            LUMINOUS_VAR_DISPATCH(spp)
        }

        SensorSample Sampler::sensor_sample(uint2 p_raster, const Filter *filter) {
            SensorSample ss;
            FilterSample fs = filter->sample(next_2d());
            ss.p_film = make_float2(p_raster) + make_float2(0.5f) + fs.p;
            ss.p_lens = next_2d();
            ss.time = next_1d();
            ss.filter_weight = fs.weight;
            return ss;
        }

        int Sampler::compute_dimension(int depth) const {
            LUMINOUS_VAR_DISPATCH(compute_dimension, depth)
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

    }
}