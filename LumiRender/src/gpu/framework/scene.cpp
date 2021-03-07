//
// Created by Zero on 2021/2/1.
//


#include "scene.h"

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

            for (const SP<const Model> &model : scene_graph->model_list) {
                for (const SP<const Mesh> &mesh : model->meshes) {
                    P.insert(P.end(), mesh->positions.begin(), mesh->positions.end());
                    N.insert(N.end(), mesh->normals.begin(), mesh->normals.end());
                    UV.insert(UV.end(), mesh->tex_coords.begin(), mesh->tex_coords.end());
                    T.insert(T.end(), mesh->triangles.begin(), mesh->triangles.end());
                }
            }

            for (const auto &instance : scene_graph->instance_list) {

            }
        }

        void Scene::build_accel() {

        }
    }
}