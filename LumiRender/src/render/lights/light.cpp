//
// Created by Zero on 2021/4/7.
//

#include "common.h"
#include "light.h"
#include "render/include/trace.h"
#include "core/refl/factory.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {


        LightType Light::type() const {
            LUMINOUS_VAR_PTR_DISPATCH(type);
        }

        LightEvalContext Light::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            LUMINOUS_VAR_PTR_DISPATCH(sample, lls, u, scene_data);
        }

        LightLiSample Light::Li(LightLiSample lls, const SceneData *data) const {
            LUMINOUS_VAR_PTR_DISPATCH(Li, lls, data);
        }

        lstd::optional<LightLiSample> Light::sample_Li(float2 u, LightLiSample lls, uint64_t traversable_handle,
                                                       const SceneData *scene_data) const {
            lls.lec = sample(&lls, u, scene_data);
            Ray ray = lls.lsc.spawn_ray_to(lls.lec);
            bool occluded = intersect_any(traversable_handle, ray);
            if (occluded) {
                return {};
            }
            lls = Li(lls, scene_data);
            float factor = lls.PDF_dir == 0 ? 0 : 1;
            lls.L *= factor;
            return lls;
        }


        Spectrum Light::MIS_sample_light(const Interaction &it, Sampler &sampler,
                                         uint64_t traversable_handle, const SceneData *scene_data) const {
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            Spectrum Ld(0.f);
            auto si = (const SurfaceInteraction &)it;
            auto bsdf = si.op_bsdf.value();
            LightLiSample lls{LightSampleContext(it)};
            auto op_lls = sample_Li(sampler.next_2d(), lls, traversable_handle, scene_data);
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

        Spectrum Light::MIS_sample_BSDF(const Interaction &it, Sampler &sampler,
                                        uint64_t traversable_handle, NEEData *NEE_data,
                                        const SceneData *data) const {
            Spectrum Ld(0.f);
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            auto si = (const SurfaceInteraction &)it;
            auto bsdf = si.op_bsdf.value();
            float uc = sampler.next_1d();
            float2 u = sampler.next_2d();
            auto bsdf_sample = bsdf.sample_f(si.wo, uc, u);
            if (bsdf_sample) {
                NEE_data->wi = bsdf_sample->wi;
                NEE_data->bsdf_val = bsdf_sample->f_val;
                NEE_data->bsdf_PDF = bsdf_sample->PDF;
                float weight = 1;
                bsdf_PDF = bsdf_sample->PDF;
                bsdf_val = bsdf_sample->f_val;
                Ray ray = si.spawn_ray(NEE_data->wi);
                HitContext hit_ctx{data};
                NEE_data->found_intersection = intersect_closest(traversable_handle, ray, &hit_ctx.hit_info);
                if (hit_ctx.is_hit() && (NEE_data->next_si = hit_ctx.compute_surface_interaction(ray)).light == this) {
                    NEE_data->next_si.update_PDF_pos(data->compute_prim_PMF(hit_ctx.hit_info));
                    Li = NEE_data->next_si.Le(-NEE_data->wi, data);
                    light_PDF = as<AreaLight>()->PDF_Li(LightSampleContext(si), LightEvalContext{NEE_data->next_si}, NEE_data->wi, data);
                } else if (!NEE_data->found_intersection && is_infinite()) {
                    Li = as<Envmap>()->on_miss(ray.direction(), hit_ctx.scene_data());
                    light_PDF = as<Envmap>()->PDF_Li(LightSampleContext(si), LightEvalContext{NEE_data->next_si}, NEE_data->wi, data);
                }
                weight = MIS_weight(bsdf_PDF, light_PDF);
                Ld = bsdf_val * Li * weight / bsdf_PDF;
            }
            return Ld;
        }

        Spectrum Light::estimate_direct_lighting(const Interaction &it, Sampler &sampler,
                                                 uint64_t traversable_handle, const SceneData *scene_data,
                                                 NEEData *NEE_data) const {

            Spectrum Ld = MIS_sample_light(it, sampler, traversable_handle, scene_data);
            return Ld + MIS_sample_BSDF(it, sampler, traversable_handle, NEE_data, scene_data);
        }

        bool Light::is_delta() const {
            LUMINOUS_VAR_PTR_DISPATCH(is_delta);
        }

        bool Light::is_infinite() const {
            LUMINOUS_VAR_PTR_DISPATCH(is_infinite);
        }

        Spectrum Light::on_miss(float3 dir, const SceneData *data) const {
            LUMINOUS_VAR_PTR_DISPATCH(on_miss, dir, data);
        }

        float Light::PDF_Li(const LightSampleContext &ctx, const LightEvalContext &p_light,
                            float3 wi, const SceneData *data) const {
            LUMINOUS_VAR_PTR_DISPATCH(PDF_Li, ctx, p_light, wi, data);
        }

        Spectrum Light::power() const {
            LUMINOUS_VAR_PTR_DISPATCH(power);
        }

        void Light::print() const {
            LUMINOUS_VAR_PTR_DISPATCH(print);
        }

        CPU_ONLY(LM_NODISCARD std::string Light::to_string() const {
            LUMINOUS_VAR_PTR_DISPATCH(to_string);
        })

        CPU_ONLY(std::pair<Light, std::vector<size_t>> Light::create(const LightConfig &config) {
            return detail::create_ptr<Light>(config);
        })

    } // luminous::render
} // luminous