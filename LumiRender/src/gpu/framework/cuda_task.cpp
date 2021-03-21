//
// Created by Zero on 2021/2/18.
//

#include "cuda_task.h"

namespace luminous {
    inline namespace gpu {

        void CUDATask::on_key(int key, int scancode, int action, int mods) {
            switch (key) {
                case 'A':
                    break;
                case 'S':
                    break;
                case 'D':
                    break;
                case 'W':
                    break;
                case 'Q':
                    break;
                case 'E':
                    break;
            }
        }

        void CUDATask::update_camera_fov_y(float val) {
            _camera.update_fov_y(val);
        }

        void CUDATask::update_camera_view(float d_yaw, float d_pitch) {
            _camera.update_yaw(d_yaw);
            _camera.update_pitch(d_pitch);
        }

        void CUDATask::update_film_resolution(int2 res) {
            auto film = _camera.film();
            film->set_resolution(res);
        }

        void CUDATask::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
            _scene = make_unique<Scene>(_device);
            _scene->convert_geometry_data(scene_graph);
            _scene->build_accel();

            _camera = SensorHandle::create(scene_graph->sensor_config);
            auto film = _camera.film();
            cout << film->to_string() << endl;
        }

        void CUDATask::render_gui() {

        }

        int2 CUDATask::resolution() {
            return _camera.film()->resolution();
        }

    }
}