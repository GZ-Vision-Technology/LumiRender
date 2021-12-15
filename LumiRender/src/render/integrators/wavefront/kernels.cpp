//
// Created by Zero on 14/10/2021.
//

#include "kernels.h"
#include "render/samplers/shader_include.h"
#include "render/light_samplers/shader_include.h"
#include "render/lights/shader_include.h"

#ifdef __CUDACC__
#define GLOBAL_PREFIX extern "C" __constant__
#else
#define GLOBAL_PREFIX static
#endif

namespace luminous {
    inline namespace render {

        GLOBAL_PREFIX RTParam *rt_param;

        CPU_ONLY(void set_rt_param(RTParam *param) {
            rt_param = param;
        })

        void generate_primary_ray(int task_id, int n_item, int y0, int sample_index,
                                  RayQueue *ray_queue, SOA<PixelSampleState> *pixel_sample_state) {

            uint res_x = rt_param->camera->resolution().x;
            uint2 pixel = make_uint2(task_id % res_x, y0 + task_id / res_x);
            pixel_sample_state->pixel[task_id] = pixel;

            Sensor *camera = rt_param->camera;
            Sampler sampler = *(rt_param->sampler);
            sampler.start_pixel_sample(pixel, sample_index, 0);
            SensorSample ss = sampler.sensor_sample(pixel, camera->filter());
            auto[weight, ray] = camera->generate_ray(ss);
            pixel_sample_state->Li[task_id] = {0.f};
            pixel_sample_state->normal[task_id] = make_float3(0.f);
            pixel_sample_state->albedo[task_id] = make_float3(0.f);
            pixel_sample_state->filter_weight[task_id] = 1.f;
            ray_queue->push_primary_ray(ray, task_id);
        }

        void generate_ray_samples(int task_id, int n_item, int sample_index,
                                  const RayQueue *ray_queue,
                                  SOA<PixelSampleState> *pixel_sample_state) {
            Sampler sampler = *(rt_param->sampler);
            RayWorkItem item = (*ray_queue)[task_id];
            uint2 pixel = pixel_sample_state->pixel[item.pixel_index];
            int dimension = sampler.compute_dimension(item.depth);
            sampler.start_pixel_sample(pixel, sample_index, dimension);

            RaySamples ray_samples;
            ray_samples.direct.uc = sampler.next_1d();
            ray_samples.direct.u = sampler.next_2d();
            ray_samples.indirect.uc = sampler.next_1d();
            ray_samples.indirect.u = sampler.next_2d();
            ray_samples.indirect.rr = sampler.next_1d();

            pixel_sample_state->ray_samples[item.pixel_index] = ray_samples;
        }

        void process_escape_ray(int task_id, int n_item,
                                EscapedRayQueue *escaped_ray_queue,
                                SOA<PixelSampleState> *pixel_sample_state) {
            if (task_id >= escaped_ray_queue->size()) {
                return;
            }
            const SceneData *scene_data = &(rt_param->scene_data);
            const LightSampler *light_sampler = scene_data->light_sampler;
            EscapedRayWorkItem item = (*escaped_ray_queue)[task_id];
            Spectrum L = pixel_sample_state->Li[item.pixel_index];

            if (item.depth == 0) {
                L += light_sampler->on_miss(item.ray_d, scene_data, item.throughput);
            } else {
                for (int i = 0; i < light_sampler->infinite_light_num(); ++i) {
                    const Light &light = light_sampler->infinite_light_at(i);
                    LightSampleContext prev_lsc = item.prev_lsc;
                    float light_select_PMF = light_sampler->PMF(prev_lsc, light);
                    LightLiSample lls{prev_lsc, normalize(item.ray_d)};
                    float bsdf_PDF = item.prev_bsdf_PDF;
                    Spectrum bsdf_val = item.prev_bsdf_val;
                    lls = light.Li(lls, scene_data);
                    float weight = MIS_weight(bsdf_PDF, lls.PDF_dir);
                    Spectrum temp_Li = item.throughput * lls.L * bsdf_val * weight / bsdf_PDF;
                    L += temp_Li;
                }
            }

            pixel_sample_state->Li[item.pixel_index] = L;
        }

        void process_emission(int task_id, int n_item,
                              HitAreaLightQueue *hit_area_light_queue,
                              SOA<PixelSampleState> *pixel_sample_state) {
            if (task_id >= hit_area_light_queue->size()) {
                return;
            }
            auto light_sampler = rt_param->scene_data.light_sampler;
            HitAreaLightWorkItem item = (*hit_area_light_queue)[task_id];
            const SceneData *scene_data = &(rt_param->scene_data);
            HitInfo hit_info = item.light_hit_info;
            HitContext hit_ctx{hit_info, scene_data};

            LightEvalContext lec = hit_ctx.compute_light_eval_context();

            const Light *light = hit_ctx.light();

            float3 wo = item.wo;
            Spectrum temp_Li = pixel_sample_state->Li[item.pixel_index];
            if (item.depth == 0) {
                Spectrum Le = light->as<AreaLight>()->radiance(lec, wo, scene_data);
                temp_Li += Le * item.throughput;
            } else {
                LightLiSample lls{item.prev_lsc, lec};
                float light_select_PMF = scene_data->light_sampler->PMF(*light);
                lls = light->Li(lls, scene_data);
                float light_PDF = lls.PDF_dir;
                float bsdf_PDF = item.prev_bsdf_PDF;
                Spectrum bsdf_val = item.prev_bsdf_val;
                float weight = MIS_weight(bsdf_PDF, light_PDF);
                Spectrum L = item.throughput * lls.L * bsdf_val * weight / bsdf_PDF;
                temp_Li += L / light_select_PMF;
            }
            pixel_sample_state->Li[item.pixel_index] = temp_Li;
        }

        void estimate_direct_lighting(int task_id, int n_item,
                                      ShadowRayQueue *shadow_ray_queue,
                                      RayQueue *next_ray_queue,
                                      MaterialEvalQueue *material_eval_queue,
                                      SOA<PixelSampleState> *pixel_sample_state) {
            if (task_id >= material_eval_queue->size()) {
                return;
            }
            MaterialEvalWorkItem mtl_item = (*material_eval_queue)[task_id];
            const SceneData *scene_data = &(rt_param->scene_data);


            HitContext hit_ctx{mtl_item.hit_info, scene_data};
            SurfaceInteraction si = hit_ctx.compute_surface_interaction(mtl_item.wo);
            BSDFWrapper bsdf = si.get_BSDF(scene_data);
            if (mtl_item.depth == 0) {
                pixel_sample_state->normal[mtl_item.pixel_index] = si.g_uvn.normal;
                pixel_sample_state->albedo[mtl_item.pixel_index] = make_float3(bsdf.base_color());
            }

            RaySamples rs = pixel_sample_state->ray_samples[mtl_item.pixel_index];

            // sample BSDF
            auto bsdf_sample = bsdf.sample_f(mtl_item.wo, rs.indirect.uc, rs.indirect.u);
            if (bsdf_sample.valid()) {
                Spectrum throughput = mtl_item.throughput * bsdf_sample.f_val / bsdf_sample.PDF;
                RayWorkItem ray_item;
                ray_item.prev_bsdf_val = bsdf_sample.f_val;
                ray_item.prev_bsdf_PDF = bsdf_sample.PDF;
                Ray new_ray = si.spawn_ray(bsdf_sample.wi);
                next_ray_queue->push_secondary_ray(new_ray, mtl_item.depth + 1, LightSampleContext(si),
                                                   throughput, bsdf_sample.PDF,
                                                   bsdf_sample.f_val, 1,
                                                   mtl_item.pixel_index);
            }

            // sample light
            const LightSampler *light_sampler = scene_data->light_sampler;
            LightSampleContext lsc{si};
            SampledLight sampled_light = light_sampler->sample(lsc, rs.direct.uc);
            const Light *light = sampled_light.light;
            LightLiSample lls{lsc};


//            auto op_lls = light->sample_Li(rs.direct.u, lls)
        }

        void add_samples(int task_id, int n_item,
                         SOA<PixelSampleState> *pixel_sample_state) {
            Sensor *camera = rt_param->camera;
            uint2 pixel = pixel_sample_state->pixel[task_id];
            Film *film = camera->film();
            Spectrum L = pixel_sample_state->Li[task_id];
            float3 normal = pixel_sample_state->normal[task_id];
            float3 albedo = pixel_sample_state->albedo[task_id];
            film->add_samples(pixel, L, albedo, normal, 1, rt_param->frame_index);
        }
    }
}