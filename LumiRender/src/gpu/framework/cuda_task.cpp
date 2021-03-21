//
// Created by Zero on 2021/2/18.
//

#include "cuda_task.h"

namespace luminous {
    inline namespace gpu {

        void CUDATask::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
            _scene = make_unique<Scene>(_device);
            _scene->convert_geometry_data(scene_graph);
            _scene->build_accel();

            _camera = SensorHandle::create(scene_graph->sensor_config);

            cout << _camera.to_string() << endl;
        }

        void CUDATask::render_gui() {

        }
    }
}