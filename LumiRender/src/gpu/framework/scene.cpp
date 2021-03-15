//
// Created by Zero on 2021/2/1.
//


#include "scene.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include "gpu/framework/jitify/jitify.hpp"

namespace luminous {
    inline namespace gpu {

        Scene::Scene(const SP<Device> &device)
                : _device(device) {
            _optix_accel = make_unique<OptixAccel>(device);

        }

        void Scene::convert_geometry_data(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("convert geometry data start!")
            uint vert_offset = 0u;
            uint tri_offset = 0u;
            for (const SP<const Model> &model : scene_graph->model_list) {
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _cpu_positions.insert(_cpu_positions.end(), mesh->positions.begin(), mesh->positions.end());
                    _cpu_normals.insert(_cpu_normals.end(), mesh->normals.begin(), mesh->normals.end());
                    _cpu_tex_coords.insert(_cpu_tex_coords.end(), mesh->tex_coords.begin(), mesh->tex_coords.end());
                    _cpu_triangles.insert(_cpu_triangles.end(), mesh->triangles.begin(), mesh->triangles.end());
                    uint vert_count = mesh->positions.size();
                    uint tri_count = mesh->triangles.size();
                    mesh->idx_in_meshes = _cpu_meshes.size();
                    _cpu_meshes.emplace_back(vert_offset, tri_offset, vert_count, tri_count);
                    vert_offset += vert_count;
                    tri_offset += tri_count;
                }
            }
            _positions = _device->allocate_buffer<float3>(_cpu_positions.size());
            _normals = _device->allocate_buffer<float3>(_cpu_normals.size());
            _tex_coords = _device->allocate_buffer<float2>(_cpu_tex_coords.size());
            _triangles = _device->allocate_buffer<TriangleHandle>(_cpu_triangles.size());
            _meshes = _device->allocate_buffer<MeshHandle>(_cpu_meshes.size());
            auto dispatcher = _device->new_dispatcher();
            _positions.upload_async(dispatcher, _cpu_positions.data());
            _normals.upload_async(dispatcher, _cpu_normals.data());
            _tex_coords.upload_async(dispatcher, _cpu_tex_coords.data());
            _triangles.upload_async(dispatcher, _cpu_triangles.data());
            _meshes.upload_async(dispatcher, _cpu_meshes.data(), _cpu_meshes.size());

            for (const SP<const ModelInstance> &instance : scene_graph->instance_list) {
                const SP<const Model> &model = scene_graph->model_list[instance->model_idx];
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _cpu_instance_to_transform_idx.push_back(_cpu_transforms.size());
                    _cpu_instance_to_mesh_idx.push_back(mesh->idx_in_meshes);
                }
                _cpu_transforms.push_back(instance->o2w.mat4x4());
            }
            _transforms = _device->allocate_buffer<float4x4>(_cpu_transforms.size());
            _instance_to_mesh_idx = _device->allocate_buffer<uint>(_cpu_instance_to_mesh_idx.size());
            _instance_to_transform_idx = _device->allocate_buffer<uint>(_cpu_instance_to_transform_idx.size());
            _transforms.upload_async(dispatcher, _cpu_transforms.data());
            _instance_to_mesh_idx.upload_async(dispatcher, _cpu_instance_to_mesh_idx.data());
            _instance_to_transform_idx.upload_async(dispatcher, _cpu_instance_to_transform_idx.data());
            dispatcher.wait();
        }

        void Scene::build_accel() {
            _optix_accel->build_bvh(_positions, _triangles, _cpu_meshes,
                                    _instance_to_mesh_idx,_cpu_transforms,
                                    _cpu_instance_to_transform_idx);
        }
    }
}