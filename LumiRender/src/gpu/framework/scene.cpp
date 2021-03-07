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

        template <typename T>
        bool are_close(T in, T out) {
            return fabs(in - out) <= 1e-5f * fabs(in);
        }

        template <typename T>
        bool test_parallel_for() {
            int n = 10000;
            T* d_out;
            cudaMalloc((void**)&d_out, n * sizeof(T));
            T val = 3.14159f;

            jitify::ExecutionPolicy policy(jitify::DEVICE);
            auto lambda = JITIFY_LAMBDA((d_out, val),
                                        d_out[i] = static_cast<decltype(val)>(i) * val);
            CUDA_CHECK(jitify::parallel_for(policy, 0, n, lambda));

            std::vector<T> h_out(n);
            cudaMemcpy(&h_out[0], d_out, n * sizeof(T), cudaMemcpyDeviceToHost);

            cudaFree(d_out);

            for (int i = 0; i < n; ++i) {
                if (!are_close(h_out[i], (T)i * val)) {
                    std::cout << h_out[i] << " != " << (T)i * val << std::endl;
                    return false;
                }
            }
            return true;
        }

        void Scene::convert_geometry_data(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("convert geometry data start!")
            vector<float3> P;
            vector<float3> N;
            vector<float2> UV;
            vector<TriangleHandle> T;
            for (const SP<const Model> &model : scene_graph->model_list) {
                for (const SP<const Mesh> &mesh : model->meshes) {
                    P.insert(P.end(), mesh->positions.begin(), mesh->positions.end());
                    N.insert(N.end(), mesh->normals.begin(), mesh->normals.end());
                    UV.insert(UV.end(), mesh->tex_coords.begin(), mesh->tex_coords.end());
                    T.insert(T.end(), mesh->triangles.begin(), mesh->triangles.end());
                }
            }
            _positions = _device->allocate_buffer<float3>(P.size());
            _normals = _device->allocate_buffer<float3>(N.size());
            _tex_coords = _device->allocate_buffer<float2>(UV.size());
            _triangles = _device->allocate_buffer<TriangleHandle>(T.size());
            auto dispatcher = _device->new_dispatcher();
            _positions.upload(dispatcher, P.data(), P.size());

//            test_parallel_for<float>();

            for (const SP<const ModelInstance> &instance : scene_graph->instance_list) {

            }
        }

        void Scene::build_accel() {

        }
    }
}