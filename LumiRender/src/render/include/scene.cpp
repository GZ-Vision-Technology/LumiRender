//
// Created by Zero on 2021/3/24.
//

#include "scene.h"

namespace luminous {
    inline namespace render {

        template<typename T>
        size_t _size_in_bytes(const vector<T> &v) {
            return v.size() * sizeof(T);
        }

        template<typename T>
        auto append(vector<T> &a, const vector<T> &b) {
            return a.insert(a.end(), b.begin(), b.end());
        }

        void Scene::convert_geometry_data(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("convert geometry data start!")
            uint vert_offset = 0u;
            uint tri_offset = 0u;
            for (const SP<const Model> &model : scene_graph->model_list) {
                for (const SP<const Mesh> &mesh : model->meshes) {
                    append(_cpu_positions, mesh->positions);
                    append(_cpu_normals, mesh->normals);
                    append(_cpu_tex_coords, mesh->tex_coords);
                    append(_cpu_triangles, mesh->triangles);

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
                    _cpu_inst_to_transform_idx.push_back(_cpu_transforms.size());
                    _cpu_inst_to_mesh_idx.push_back(mesh->idx_in_meshes);
                }
                _cpu_transforms.push_back(instance->o2w.mat4x4());
            }
            shrink_to_fit();
        }

        size_t Scene::size_in_bytes() const {
            size_t ret = _size_in_bytes(_cpu_triangles);
            ret += _size_in_bytes(_cpu_tex_coords);
            ret += _size_in_bytes(_cpu_positions);
            ret += _size_in_bytes(_cpu_normals);

            ret += _size_in_bytes(_cpu_meshes);
            ret += _size_in_bytes(_cpu_transforms);
            ret += _size_in_bytes(_cpu_inst_to_mesh_idx);
            ret += _size_in_bytes(_cpu_inst_to_transform_idx);

            return ret;
        }

        void Scene::clear() {
            _cpu_triangles.clear();
            _cpu_tex_coords.clear();
            _cpu_positions.clear();
            _cpu_normals.clear();
            _cpu_meshes.clear();
            _cpu_transforms.clear();
            _cpu_inst_to_mesh_idx.clear();
            _cpu_inst_to_transform_idx.clear();
        }

        void Scene::shrink_to_fit() {
            _cpu_triangles.shrink_to_fit();
            _cpu_tex_coords.shrink_to_fit();
            _cpu_positions.shrink_to_fit();
            _cpu_normals.shrink_to_fit();
            _cpu_meshes.shrink_to_fit();
            _cpu_transforms.shrink_to_fit();
            _cpu_inst_to_mesh_idx.shrink_to_fit();
            _cpu_inst_to_transform_idx.shrink_to_fit();
        }

    }
}