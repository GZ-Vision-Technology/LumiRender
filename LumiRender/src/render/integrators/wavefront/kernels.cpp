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
            SensorSample ss = sampler.sensor_sample(pixel);
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
            L += light_sampler->on_miss(item.ray_d, scene_data, item.throughput);
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

            Spectrum Le = item.light->as<AreaLight>()->radiance(item.uv, item.ng, item.wo, &rt_param->scene_data);
            Spectrum Li{0.f};
            Spectrum temp_Li = pixel_sample_state->Li[item.pixel_index];
            Li = Le * item.throughput;
            if (item.depth != 0) {
                float light_select_PMF = light_sampler->PMF(*item.light);
//                float light_PDF = item.light->PDF_Li(item.p)
            }


            pixel_sample_state->Li[item.pixel_index] = temp_Li;
        }

        void estimate_direct_lighting(int task_id, int n_item,
                                      ShadowRayQueue *shadow_ray_queue,
                                      RayQueue *next_ray_queue,
                                      MaterialEvalQueue *material_eval_queue) {
            if (task_id >= material_eval_queue->size()) {
                return;
            }
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