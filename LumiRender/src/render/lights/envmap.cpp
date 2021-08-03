//
// Created by Zero on 2021/7/28.
//

#include "envmap.h"


namespace luminous {
    inline namespace render {

        void Envmap::preprocess(const Scene *scene) {

        }

        LightLiSample Envmap::Li(LightLiSample lls) const {
            return LightLiSample();
        }

        SurfaceInteraction Envmap::sample(float2 u, const HitGroupData *hit_group_data) const {
            return SurfaceInteraction();
        }

        float Envmap::PDF_Li(const Interaction &p_ref, const SurfaceInteraction &p_light) const {
            return 0;
        }

        Spectrum Envmap::power() const {
            // todo
            return luminous::Spectrum(1.f);
        }

        void Envmap::print() const {
            printf("type:Envmap\n");
        }

        CPU_ONLY(std::vector<float> Envmap::create_distribution(const Image &image) {
            auto width = image.width();
            auto height = image.height();
            std::vector<float> ret(image.pixel_num(), 0);
            auto func = [&](const std::byte *pixel, index_t idx) {
                float f = 0;
                float v = idx / width + 0.5f;
                float theta = v / height;
                float sinTheta = std::sin(Pi * theta);
                switch (image.pixel_format()) {
                    case PixelFormat::RGBA32F:{
                        float4 val = *(reinterpret_cast<const float4*>(pixel));
                        f = luminance(val);
                        break;
                    }
                    case PixelFormat::RGBA8U:{
                        uchar4 val = *(reinterpret_cast<const uchar4*>(pixel));
                        float4 f4 = make_float4(val) / 255.f;
                        f = luminance(f4);
                        break;
                    }
                    default:
                        break;
                }
                ret[idx] = f * sinTheta;
            };
            image.for_each(func);
            return ret;
        })

        CPU_ONLY(Envmap Envmap::create(const LightConfig &config) {
            return Envmap(config.texture_config.tex_idx, config.o2w_config.create(),
                          config.distribution_idx);
        })
    }
}