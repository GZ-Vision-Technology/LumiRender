//
// Created by Zero on 02/09/2021.
//

#include "integrator.h"
#include "render/scene/scene.h"
#include "render/scene/cpu_scene.h"
#include "render/scene/gpu_scene.h"


namespace luminous {
    inline namespace render {
        const SceneData* Integrator::scene_data() const { return _scene->scene_data();}

        void Integrator::init(const std::shared_ptr<SceneGraph> &scene_graph) {
            if (_device->is_cpu()) {
                _scene = std::make_shared<CPUScene>(_device, _context);
            } else {
                _scene = std::make_shared<GPUScene>(_device, _context);
            }
            _max_depth = scene_graph->integrator_config.max_depth;
            _rr_threshold = scene_graph->integrator_config.rr_threshold;
            _scene->init(scene_graph);
            _camera.push_back(Sensor::create(scene_graph->sensor_config));
            _sampler.push_back(Sampler::create(scene_graph->sampler_config));

            _sampler_p.init(1);
            _sampler_p.add_element(scene_graph->sampler_config);

            init_on_device();
            LUMINOUS_INFO(get_arena().description());
        }
    }
}