//
// Created by Zero on 2021/4/9.
//

#include "common.h"
#include "light_sampler.h"
#include "render/include/creator.h"
#include "render/lights/light.h"
#include "render/include/trace.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        void LightSampler::set_lights(BufferView<const Light> lights) {
            LUMINOUS_VAR_PTR_DISPATCH(set_lights, lights);
        }

        void LightSampler::set_infinite_lights(BufferView<const Light> lights) {
            LUMINOUS_VAR_PTR_DISPATCH(set_infinite_lights, lights);
        }

        size_t LightSampler::light_num() const {
            LUMINOUS_VAR_PTR_DISPATCH(light_num);
        }

        size_t LightSampler::infinite_light_num() const {
            LUMINOUS_VAR_PTR_DISPATCH(infinite_light_num);
        }

        SampledLight LightSampler::sample(float u) const {
            LUMINOUS_VAR_PTR_DISPATCH(sample, u);
        }

        CPU_ONLY(LM_NODISCARD std::string LightSampler::to_string() const {
            LUMINOUS_VAR_PTR_DISPATCH(to_string);
        })

        SampledLight LightSampler::sample(const LightSampleContext &ctx, float u) const {
            LUMINOUS_VAR_PTR_DISPATCH(sample, ctx, u);
        }

        Spectrum LightSampler::MIS_sample_light(const SurfaceInteraction &si, const BSDFWrapper &bsdf, Sampler &sampler,
                                                uint64_t traversable_handle, const SceneData *scene_data) const {
            auto sampled_light = sample(LightSampleContext(si), sampler.next_1d());
            float light_PDF = 0, bsdf_PDF = 0;
            Spectrum bsdf_val(0.f), Li(0.f);
            Spectrum Ld(0.f);
            LightLiSample lls{LightSampleContext(si)};
            lls = sampled_light.light->sample_Li(sampler.next_2d(), lls, traversable_handle, scene_data);
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
            Ld = Ld / sampled_light.PMF;
            DCHECK(!has_invalid(Ld));
            return select(lls.valid() && lls.has_contribution(), Ld, Spectrum{0.f});
        }

        Spectrum LightSampler::MIS_sample_BSDF(const SurfaceInteraction &si, const BSDFWrapper &bsdf, Sampler &sampler,
                                               uint64_t traversable_handle, PathVertex *vertex,
                                               const SceneData *data) const {
            Spectrum Ld(0.f);
            float light_PDF = 0, bsdf_PDF = 0;
            float light_PMF = 1.f;
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
                if (hit_ctx.is_hit() && (vertex->next_si = hit_ctx.compute_surface_interaction(ray)).light) {
                    vertex->next_si.update_PDF_pos(data->compute_prim_PMF(hit_ctx.hit_info));
                    Li = vertex->next_si.Le(-vertex->wi, data);
                    light_PMF = PMF(*vertex->next_si.light);
                    light_PDF = vertex->next_si.light->PDF_Li(LightSampleContext(si), LightEvalContext{vertex->next_si},
                                                        vertex->wi, data);
                } else if (!vertex->found_intersection && infinite_light_num() != 0) {
                    const Light &light = infinite_light_at(0);
                    light_PMF = PMF(light);
                    LightLiSample lls(LightSampleContext(si), ray.direction());
                    lls = light.Li(lls, data);
                    Li = lls.L;
                    light_PDF = light.PDF_Li(LightSampleContext(si), LightEvalContext{vertex->next_si}, vertex->wi, data);
                }
                float weight = bsdf_sample.is_specular() ? 1 : MIS_weight(bsdf_PDF, light_PDF);
                Ld = bsdf_val * Li * weight / bsdf_PDF / light_PMF;
            }
            DCHECK(!has_invalid(Ld));
            return Ld;
        }

        Spectrum LightSampler::estimate_direct_lighting(const SurfaceInteraction &si, Sampler &sampler,
                                                        uint64_t traversable_handle,
                                                        const SceneData *scene_data,
                                                        PathVertex *vertex) const {
            auto bsdf = si.compute_BSDF(scene_data);
            Spectrum Ld = MIS_sample_light(si, bsdf, sampler, traversable_handle, scene_data);
            Ld += MIS_sample_BSDF(si, bsdf, sampler, traversable_handle, vertex, scene_data);
            return Ld;
        }

        const Light &LightSampler::light_at(uint idx) const {
            LUMINOUS_VAR_PTR_DISPATCH(light_at, idx);
        }

        const Light &LightSampler::infinite_light_at(uint idx) const {
            LUMINOUS_VAR_PTR_DISPATCH(infinite_light_at, idx);
        }

        float LightSampler::PMF(const Light &light) const {
            LUMINOUS_VAR_PTR_DISPATCH(PMF, light);
        }

        BufferView<const Light> LightSampler::lights() const {
            LUMINOUS_VAR_PTR_DISPATCH(lights);
        }

        BufferView<const Light> LightSampler::infinite_lights() const {
            LUMINOUS_VAR_PTR_DISPATCH(infinite_lights);
        }

        float LightSampler::PMF(const LightSampleContext &ctx, const Light &light) const {
            LUMINOUS_VAR_PTR_DISPATCH(PMF, ctx, light);
        }

        Spectrum LightSampler::on_miss(float3 dir, const SceneData *scene_data, Spectrum throughput) const {
            Spectrum L{0.f};
            BufferView<const Light> lights = infinite_lights();
            for (auto &light : lights) {
                L += throughput * light.on_miss(dir, scene_data);
            }
            return L;
        }

        CPU_ONLY(std::pair<LightSampler, std::vector<size_t>> LightSampler::create(const LightSamplerConfig &config) {
            return detail::create_ptr<LightSampler>(config);
        })

    } // luminous::render
} // luminous