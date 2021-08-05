//
// Created by Zero on 2021/7/28.
//

#include "envmap.h"


namespace luminous {
    inline namespace render {

        LightLiSample Envmap::Li(LightLiSample lls) const {
            return LightLiSample();
        }

        SurfaceInteraction Envmap::sample(float2 u, const SceneData *scene_data) const {
            return SurfaceInteraction();
        }

        Spectrum Envmap::on_miss(Ray ray, const SceneData *data) const {
            const Texture &tex = data->textures[_tex_idx];
            float3 d = normalize(_w2o.apply_vector(ray.direction()));
            float2 uv = make_float2(spherical_phi(d) * inv2Pi, spherical_theta(d) * invPi);
            float4 L = tex.eval(uv);
            return Spectrum(make_float3(L));
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
            image.for_each_pixel(func);
            return ret;
        })

        CPU_ONLY(Envmap Envmap::create(const LightConfig &config) {
            Transform o2w = config.o2w_config.create();
            Transform rotate_x = Transform::rotation_x(90);
            o2w = o2w * rotate_x;
            return Envmap(config.texture_config.tex_idx, o2w.inverse(),
                          config.distribution_idx, config.scene_box);
        })
    }
}