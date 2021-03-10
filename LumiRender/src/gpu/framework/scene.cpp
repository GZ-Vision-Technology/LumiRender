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
            vector<float3> P;
            vector<float3> N;
            vector<float2> UV;
            vector<TriangleHandle> T;
            vector<MeshHandle> mesh_list;
            uint vert_offset = 0u;
            uint tri_offset = 0u;
            for (const SP<const Model> &model : scene_graph->model_list) {
                for (const SP<const Mesh> &mesh : model->meshes) {
                    P.insert(P.end(), mesh->positions.begin(), mesh->positions.end());
                    N.insert(N.end(), mesh->normals.begin(), mesh->normals.end());
                    UV.insert(UV.end(), mesh->tex_coords.begin(), mesh->tex_coords.end());
                    T.insert(T.end(), mesh->triangles.begin(), mesh->triangles.end());
                    uint vert_count = mesh->positions.size();
                    uint tri_count = mesh->triangles.size();
                    mesh->idx_in_meshes = mesh_list.size();
                    mesh_list.emplace_back(vert_offset, tri_offset, vert_count, tri_count);
                    vert_offset += vert_count;
                    tri_offset += tri_count;
                }
            }
            _positions = _device->allocate_buffer<float3>(P.size());
            _normals = _device->allocate_buffer<float3>(N.size());
            _tex_coords = _device->allocate_buffer<float2>(UV.size());
            _triangles = _device->allocate_buffer<TriangleHandle>(T.size());
            _meshes = _device->allocate_buffer<MeshHandle>(mesh_list.size());
            auto dispatcher = _device->new_dispatcher();
            _positions.upload_async(dispatcher, P.data(), P.size());
            _normals.upload_async(dispatcher, N.data(), N.size());
            _tex_coords.upload_async(dispatcher, UV.data(), UV.size());
            _triangles.upload_async(dispatcher, T.data(), T.size());
            _meshes.upload_async(dispatcher, mesh_list.data(), mesh_list.size());

            vector<float4x4> transforms;
            vector<uint> inst_to_mesh_idx;
            vector<uint> inst_tsf_idx;
            for (const SP<const ModelInstance> &instance : scene_graph->instance_list) {
                const SP<const Model> &model = scene_graph->model_list[instance->model_idx];
                for (const SP<const Mesh> &mesh : model->meshes) {
                    inst_tsf_idx.push_back(transforms.size());
                    inst_to_mesh_idx.push_back(mesh->idx_in_meshes);
                }
                transforms.push_back(instance->o2w.mat4x4());
            }
            _transforms = _device->allocate_buffer<float4x4>(transforms.size());
            _instance_to_mesh_idx = _device->allocate_buffer<uint>(inst_to_mesh_idx.size());
            _instance_transform_idx = _device->allocate_buffer<uint>(inst_tsf_idx.size());
            _transforms.upload_async(dispatcher, transforms.data(), transforms.size());
            _instance_to_mesh_idx.upload_async(dispatcher, inst_to_mesh_idx.data(), inst_to_mesh_idx.size());
            _instance_transform_idx.upload_async(dispatcher, inst_tsf_idx.data(), inst_tsf_idx.size());
            dispatcher.wait();
        }

        void Scene::build_accel() {

        }
    }
}