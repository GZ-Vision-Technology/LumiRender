//
// Created by Zero on 2021/4/7.
//

#include "light.h"
#include "render/include/creator.h"
//#include "render/include/interaction.cpp"

namespace luminous {
    inline namespace render {


        LightType Light::type() const {
            LUMINOUS_VAR_DISPATCH(type);
        }

        SurfaceInteraction Light::sample(float2 u, const HitGroupData *hit_group_data) const {
            LUMINOUS_VAR_DISPATCH(sample, u, hit_group_data);
        }

        LightLiSample Light::Li(LightLiSample lls) const {
            LUMINOUS_VAR_DISPATCH(Li, lls);
        }

        lstd::optional<LightLiSample> Light::sample_Li(float2 u, LightLiSample lls, uint64_t traversable_handle,
                                                       const HitGroupData *hit_group_data) const {
            LUMINOUS_VAR_DISPATCH(sample_Li, u, lls, traversable_handle, hit_group_data);
        }

        Spectrum Light::estimate_direct_lighting(const SurfaceInteraction &si, const BSDF &bsdf, Sampler &sampler,
                                                 uint64_t traversable_handle, const HitGroupData *hit_group_data,
                                                 float3 *wi, Spectrum *bsdf_ei) const {
            *wi = make_float3(0.f);
            *bsdf_ei = Spectrum(0.f);
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            Spectrum Ld(0.f);
            LightLiSample lls;
            lls.p_ref = (const Interaction &) si;
            auto op_lls = sample_Li(sampler.next_2d(), lls, traversable_handle, hit_group_data);
            if (op_lls && op_lls->has_contribution()) {
                bsdf_val = bsdf.eval(si.wo, op_lls->wi);
                bsdf_PDF = bsdf.PDF(si.wo, op_lls->wi);
                Li = lls.L;
                light_PDF = lls.PDF_dir;
                if (bsdf_val.not_black() && bsdf_PDF != 0) {
                    Ray ray = si.spawn_ray_to(lls.p_light);
                    bool occluded = intersect_any(traversable_handle, ray);
                    if (occluded) {
                        Li = 0;
                    }
                    if (Li.not_black()) {
                        if (is_delta()) {
                            Ld += bsdf_val * Li / light_PDF;
                        } else {
                            float weight = MIS_weight(light_PDF, bsdf_PDF);
                            Ld += bsdf_val * Li * weight / light_PDF;
                        }
                    }
                }
            }

            auto bsdf_sample = bsdf.sample_f(si.wo, sampler.next_1d(), sampler.next_2d());

            if (bsdf_sample) {
                *wi = bsdf_sample->wi;
                *bsdf_ei = bsdf_sample->f_val / bsdf_sample->PDF;

                if (!is_delta()) {
                    bsdf_PDF = bsdf_sample->PDF;
                    bsdf_val = bsdf_sample->f_val;
                    Ray ray = si.spawn_ray(*wi);
                    PerRayData prd;
                    intersect_closest(traversable_handle, ray, &prd);
                    SurfaceInteraction light_si;
                    if (prd.is_hit() && (light_si = prd.get_surface_interaction()).light == this) {
                        light_PDF = PDF_dir(si, light_si);
                        float weight = MIS_weight(bsdf_PDF, light_PDF);
                        Li = light_si.Le(-*wi);
                        Ld += bsdf_val * Li * weight / bsdf_PDF;
                    }
                }
            }
            return Ld;
        }

        bool Light::is_delta() const {
            LUMINOUS_VAR_DISPATCH(is_delta);
        }

        float Light::PDF_dir(const Interaction &ref_p, const SurfaceInteraction &p_light) const {
            LUMINOUS_VAR_DISPATCH(PDF_dir, ref_p, p_light);
        }

        Spectrum Light::power() const {
            LUMINOUS_VAR_DISPATCH(power);
        }

        Light Light::create(const LightConfig &config) {
            return detail::create<Light>(config);
        }

    } // luminous::render
} // luminous