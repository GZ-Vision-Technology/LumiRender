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

        LightLiSample Light::sample_Li(float2 u, LightLiSample lls, uint64_t traversable_handle,
                                                       const SceneData *scene_data) const {
            lls.lec = sample(&lls, u, scene_data);
            Ray ray = lls.lsc.spawn_ray_to(lls.lec);
            bool occluded = intersect_any(traversable_handle, ray);
            if (occluded) {
                return {};
            }
            lls = Li(lls, scene_data);
            lls.update_Li();
            return lls;
        }


        Spectrum Light::MIS_sample_light(const SurfaceInteraction &si, const BSDFWrapper &bsdf, Sampler &sampler,
                                         uint64_t traversable_handle, const SceneData *scene_data) const {
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            Spectrum Ld(0.f);
            LightLiSample lls{LightSampleContext(si)};
            lls = sample_Li(sampler.next_2d(), lls, traversable_handle, scene_data);
            if (lls.valid() && lls.has_contribution()) {
                bsdf_val = bsdf.eval(si.wo, lls.wi);
                bsdf_PDF = bsdf.PDF(si.wo, lls.wi);
                Li = lls.L;
                light_PDF = lls.PDF_dir;
                if (bsdf_val.not_black() && bsdf_PDF != 0) {
                    if (Li.not_black()) {
                        float weight = MIS_weight(light_PDF, bsdf_PDF);
                        Ld = bsdf_val * Li * weight / light_PDF;
                    }
                }
            }
            DCHECK(!has_invalid(Ld));
            return Ld;
        }

        Spectrum Light::MIS_sample_BSDF(const SurfaceInteraction &si, const BSDFWrapper &bsdf,
                                        Sampler &sampler, uint64_t traversable_handle, PathVertex *vertex,
                                        const SceneData *data) const {
            Spectrum Ld(0.f);
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            float uc = sampler.next_1d();
            float2 u = sampler.next_2d();
            auto bsdf_sample = bsdf.sample_f(si.wo, uc, u);
            if (bsdf_sample.valid()) {
                vertex->wi = bsdf_sample.wi;
                vertex->bsdf_val = bsdf_sample.f_val;
                vertex->bsdf_PDF = bsdf_sample.PDF;
                vertex->bxdf_flags = bsdf_sample.flags;
                vertex->eta = bsdf_sample.eta;
                vertex->albedo = bsdf_sample.albedo;
                bsdf_PDF = bsdf_sample.PDF;
                bsdf_val = bsdf_sample.f_val;
                Ray ray = si.spawn_ray(vertex->wi);
                HitContext hit_ctx{data};
                vertex->found_intersection = intersect_closest(traversable_handle, ray, &hit_ctx.hit_info);
                if (hit_ctx.is_hit() && (vertex->next_si = hit_ctx.compute_surface_interaction(ray)).light == this) {
                    vertex->next_si.update_PDF_pos(data->compute_prim_PMF(hit_ctx.hit_info));
                    Li = vertex->next_si.Le(-vertex->wi, data);
                    light_PDF = as<AreaLight>()->PDF_Li(LightSampleContext(si), LightEvalContext{vertex->next_si},
                                                        vertex->wi, data);
                } else if (!vertex->found_intersection && is_infinite()) {
                    Li = as<Envmap>()->on_miss(ray.direction(), hit_ctx.scene_data());
                    light_PDF = as<Envmap>()->PDF_Li(LightSampleContext(si), LightEvalContext{vertex->next_si},
                                                     vertex->wi, data);
                }
                float weight = bsdf_sample.is_specular() ? 1 : MIS_weight(bsdf_PDF, light_PDF);
                Ld = bsdf_val * Li * weight / bsdf_PDF;
            }
            DCHECK(!has_invalid(Ld));
            return Ld;
        }

        Spectrum Light::estimate_direct_lighting(const SurfaceInteraction &si, Sampler &sampler,
                                                 uint64_t traversable_handle, const SceneData *scene_data,
                                                 PathVertex *vertex) const {
            auto bsdf = si.compute_BSDF(scene_data);
            Spectrum Ld = MIS_sample_light(si, bsdf, sampler, traversable_handle, scene_data);
            return Ld + MIS_sample_BSDF(si, bsdf, sampler, traversable_handle, vertex, scene_data);
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