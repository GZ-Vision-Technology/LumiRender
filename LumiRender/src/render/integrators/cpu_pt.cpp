//
// Created by Zero on 2021/5/29.
//

#include "cpu_pt.h"
#include "render/integrators/pt_func.h"
#include "render/scene/cpu_scene.h"
#include "render/sensors/common.h"
#include "render/sensors/sensor.h"
#include "util/progressreporter.h"

using std::cout;

using std::endl;

namespace luminous {
    inline namespace cpu {

        CPUPathTracer::CPUPathTracer(Device *device, Context *context)
                : Integrator(device, context) {

        }

        void CPUPathTracer::init(const SP<SceneGraph> &scene_graph) {
            Integrator::init(scene_graph);
            _scene->init_accel<EmbreeAccel>();
        }

        void CPUPathTracer::update() {
            _frame_index = 0;
        }

        void CPUPathTracer::render(int frame_num, ProgressReporter *progressor) {
            const uint tile_size = 16;
            uint2 res = _camera->resolution();
            for (int i = 0; i < frame_num; ++i) {
                tiled_for_2d(res, make_uint2(tile_size), [&](uint2 pixel, uint tid) {
                    Film *film = _camera->film();
                    Sampler sampler = *_sampler.data();
                    sampler.start_pixel_sample(pixel, _frame_index, 0);
                    auto ss = sampler.sensor_sample(pixel, _camera->filter());
                    auto[weight, ray] = _camera->generate_ray(ss);
                    uint spp = sampler.spp();
                    PixelInfo pixel_info;
                    bool debug = all(pixel == debug_pixel);
                    for (int i = 0; i < spp; ++i) {
                        pixel_info += luminous::render::path_tracing(ray, scene<CPUScene>()->scene_handle(), sampler,
                                                                     _min_depth, _max_depth, _rr_threshold,
                                                                     scene<CPUScene>()->scene_data_host_ptr(), debug);
                    }
                    pixel_info /= float(spp);
                    film->add_sample(pixel, pixel_info, weight, _frame_index);
                });
                if(progressor) progressor->update(1);
                ++_frame_index;
            }
        }
    }
}