//
// Created by Zero on 2021/2/1.
//


#include "gpu_scene.h"
#include <cassert>
#include <cmath>
#include <iostream>

namespace luminous {
    inline namespace gpu {

        GPUScene::GPUScene(const SP<Device> &device)
                : _device(device) {
            _optix_accel = make_unique<OptixAccel>(device);
        }

        void GPUScene::create_device_memory() {
            {
                // instance data
                _inst_to_mesh_idx.reset(_cpu_inst_to_mesh_idx, _device);
                _inst_to_transform_idx.reset(_cpu_inst_to_transform_idx, _device);
                _transforms.reset(_cpu_transforms, _device);
            }
            {
                // mesh data
                _meshes.reset(_cpu_meshes, _device);
                _positions.reset(_cpu_positions, _device);
                _tex_coords.reset(_cpu_tex_coords, _device);
                _triangles.reset(_cpu_triangles, _device);
            }
            clear_host();
        }

        void GPUScene::synchronize_to_gpu() {
            {
                // instance data
                _inst_to_mesh_idx.synchronize_to_gpu();
                _inst_to_transform_idx.synchronize_to_gpu();
                _transforms.synchronize_to_gpu();
            }
            {
                // mesh data
                _meshes.synchronize_to_gpu();
                _positions.synchronize_to_gpu();
                _tex_coords.synchronize_to_gpu();
                _triangles.synchronize_to_gpu();
            }
        }

        void GPUScene::init(const SP<SceneGraph> &scene_graph) {
            convert_geometry_data(scene_graph);
            create_device_memory();
            synchronize_to_gpu();
            build_accel();
        }

        void GPUScene::build_accel() {

            _optix_accel->build_bvh(_positions.device_buffer(), _triangles.device_buffer(),
                                    _cpu_meshes,
                                    _cpu_inst_to_mesh_idx, _cpu_transforms,
                                    _cpu_inst_to_transform_idx);
            cout << _optix_accel->description() << endl;
        }
    }
}