//
// Created by Zero on 2021/3/24.
//

#include "scene.h"

namespace luminous {
    inline namespace render {

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

            for (const SP<const ModelInstance> &instance : scene_graph->instance_list) {
                const SP<const Model> &model = scene_graph->model_list[instance->model_idx];
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _cpu_instance_to_transform_idx.push_back(_cpu_transforms.size());
                    _cpu_instance_to_mesh_idx.push_back(mesh->idx_in_meshes);
                }
                _cpu_transforms.push_back(instance->o2w.mat4x4());
            }
        }
    }
}