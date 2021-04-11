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
                : _device(device) {}

        void GPUScene::create_device_memory() {
            {
                // instance data
                _inst_to_mesh_idx.reset(move(_cpu_inst_to_mesh_idx), _device);
                _inst_to_transform_idx.reset(move(_cpu_inst_to_transform_idx), _device);
                _transforms.reset(move(_cpu_transforms), _device);
            }
            {
                // mesh data
                _meshes.reset(move(_cpu_meshes), _device);
                _positions.reset(move(_cpu_positions), _device);
                _tex_coords.reset(move(_cpu_tex_coords), _device);
                _triangles.reset(move(_cpu_triangles), _device);
                _normals.reset(move(_cpu_normals), _device);
            }
            {
                // other
                _lights.reset(move(_cpu_lights), _device);
            }
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
                _normals.synchronize_to_gpu();
            }
            {
                // other data
                _lights.synchronize_to_gpu();
            }
        }

        void GPUScene::init_accel() {
            _optix_accel = make_unique<OptixAccel>(_device, this);
            build_accel();
        }

        void GPUScene::build_emission_distribute() {

        }

        void GPUScene::init(const SP<SceneGraph> &scene_graph) {
            convert_data(scene_graph);
            create_device_memory();
            synchronize_to_gpu();
            init_accel();
        }

        void GPUScene::build_accel() {
            _optix_accel->build_bvh(_positions.device_buffer(),
                                    _triangles.device_buffer(),
                                    _meshes.vector(),
                                    _inst_to_mesh_idx.vector(),
                                    _transforms.vector(),
                                    _inst_to_transform_idx.vector());
            cout << _optix_accel->description() << endl;
            cout << description() << endl;
        }

        size_t GPUScene::size_in_bytes() const {
            size_t ret = 0u;
            ret += _inst_to_mesh_idx.size_in_bytes();
            ret += _inst_to_transform_idx.size_in_bytes();
            ret += _transforms.size_in_bytes();
            ret += _meshes.size_in_bytes();
            ret += _positions.size_in_bytes();
            ret += _normals.size_in_bytes();
            ret += _tex_coords.size_in_bytes();
            ret += _triangles.size_in_bytes();
            ret += _lights.size_in_bytes();
            return ret;
        }

        std::string GPUScene::description() const {
            float size_in_MB = float(size_in_bytes()) / sqr(1024);

            return string_printf("scene data occupy %f MB, instance triangle is %u,"
                                 " instance vertices is %u, light num is %u",
                                 size_in_MB,
                                 _inst_triangle_num,
                                 _inst_vertices_num,
                                 _lights.size());
        }

    } // luminous::gpu
} // luminous