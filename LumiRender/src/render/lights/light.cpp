//
// Created by Zero on 2021/4/7.
//

#include "light.h"
#include "render/include/creator.h"
#include "render/include/trace.h"


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
            lls.p_light = sample(u, hit_group_data);
            Ray ray = lls.p_ref.spawn_ray_to(lls.p_light);
            bool occluded = intersect_any(traversable_handle, ray);
            if (occluded) {
                return {};
            }
            lls.p_light.wo = normalize(-ray.direction());
            lls = Li(lls);
            return lls;
        }


        Spectrum Light::MIS_sample_light(const SurfaceInteraction &si, Sampler &sampler,
                                         uint64_t traversable_handle, const HitGroupData *hit_group_data) const {
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            Spectrum Ld(0.f);
            LightLiSample lls;
            auto bsdf = si.op_bsdf.value();
            lls.p_ref = (const Interaction &) si;
            auto op_lls = sample_Li(sampler.next_2d(), lls, traversable_handle, hit_group_data);
            if (op_lls && op_lls->has_contribution()) {
                bsdf_val = bsdf.eval(si.wo, op_lls->wi);
                bsdf_PDF = bsdf.PDF(si.wo, op_lls->wi);
                Li = op_lls->L;
                light_PDF = op_lls->PDF_dir;
                if (bsdf_val.not_black() && bsdf_PDF != 0) {
                    if (Li.not_black()) {
                        if (is_delta()) {
                            Ld = bsdf_val * Li / light_PDF;
                        } else {
                            float weight = MIS_weight(light_PDF, bsdf_PDF);
                            Ld = bsdf_val * Li * weight / light_PDF;
                        }
                    }
                }
            }
            return Ld;
        }

        Spectrum Light::MIS_sample_BSDF(const SurfaceInteraction &si, Sampler &sampler,
                                        uint64_t traversable_handle,
                                        NEEData *NEE_data) const {
            Spectrum Ld(0.f);
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            auto bsdf = si.op_bsdf.value();
            auto bsdf_sample = bsdf.sample_f(si.wo, sampler.next_1d(), sampler.next_2d());
            if (bsdf_sample) {
                NEE_data->wi = bsdf_sample->wi;
                NEE_data->bsdf_val = bsdf_sample->f_val;
                NEE_data->bsdf_PDF = bsdf_sample->PDF;
                if (!is_delta()) {
                    bsdf_PDF = bsdf_sample->PDF;
                    bsdf_val = bsdf_sample->f_val;
                    Ray ray = si.spawn_ray(NEE_data->wi);
                    PerRayData prd;
                    NEE_data->found_intersection = intersect_closest(traversable_handle, ray, &prd);
                    if (prd.is_hit() && (NEE_data->next_si = prd.compute_surface_interaction(ray)).light == this) {
                        NEE_data->next_si.PDF_pos = get<AreaLight>()->inv_area();
                        light_PDF = PDF_dir(si, NEE_data->next_si);
                        float weight = MIS_weight(bsdf_PDF, light_PDF);
                        Li = NEE_data->next_si.Le(-NEE_data->wi);
                        Ld = bsdf_val * Li * weight / bsdf_PDF;
                    }
                }
            }
            return Ld;
        }

        Spectrum Light::estimate_direct_lighting(const SurfaceInteraction &si, Sampler &sampler,
                                                 uint64_t traversable_handle, const HitGroupData *hit_group_data,
                                                 NEEData *NEE_data) const {

            Spectrum Ld = MIS_sample_light(si, sampler, traversable_handle, hit_group_data);
            return Ld + MIS_sample_BSDF(si, sampler, traversable_handle, NEE_data);
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

        void Light::print() const {
            LUMINOUS_VAR_DISPATCH(print);
        }

        CPU_ONLY(Light Light::create(const LightConfig &config) {
            return detail::create<Light>(config);
        })

    } // luminous::render
} // luminous