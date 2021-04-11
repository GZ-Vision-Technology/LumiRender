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
            return a.insert(a.cend(), b.cbegin(), b.cend());
        }

        void Scene::load_lights(const vector<LightConfig> &light_configs) {
            _cpu_lights.reserve(light_configs.size());
            for (const auto &lc : light_configs) {
                _cpu_lights.push_back(Light::create(lc));
            }
        }

        void Scene::preprocess_meshes() {
            auto process_mesh = [&](MeshHandle mesh) {
                if (mesh.distribute_idx == -1) {
                    return;
                }
                uint start = mesh.triangle_offset;
                uint end = start + mesh.triangle_count;
                vector<float> areas;
                areas.reserve(mesh.triangle_count);
                const float3 *pos = &_cpu_positions[mesh.vertex_offset];
                for (int i = start; i < end; ++i) {
                    TriangleHandle tri = _cpu_triangles[i];
                    float3 p0 = pos[tri.i];
                    float3 p1 = pos[tri.j];
                    float3 p2 = pos[tri.k];
                    float area = triangle_area(p0, p1, p2);
                    areas.push_back(area);
                }
                auto builder = Distribute1D::create_builder(move(areas));
                _emission_distribute_builders.push_back(builder);
            };

            for (const auto &mesh : _cpu_meshes) {
                process_mesh(mesh);
            }
        }

        void Scene::convert_data(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("convert scene data start!")
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

            int distribute_idx = 0;
            for (const SP<const ModelInstance> &instance : scene_graph->instance_list) {
                const SP<const Model> &model = scene_graph->model_list[instance->model_idx];
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _cpu_inst_to_transform_idx.push_back(_cpu_transforms.size());
                    if (!instance->emission.is_zero()) {
                        LightConfig lc;
                        lc.emission = instance->emission;
                        lc.type = "AreaLight";
                        lc.instance_idx = _cpu_inst_to_mesh_idx.size();
                        scene_graph->light_configs.push_back(lc);
                        MeshHandle &mesh_handle = _cpu_meshes[mesh->idx_in_meshes];
                        if (mesh_handle.distribute_idx == -1) {
                            mesh_handle.distribute_idx = distribute_idx++;
                        }
                    }
                    _cpu_inst_to_mesh_idx.push_back(mesh->idx_in_meshes);

                    _inst_triangle_num += mesh->triangles.size();
                    _inst_vertices_num += mesh->positions.size();
                }
                _cpu_transforms.push_back(instance->o2w.mat4x4());
            }
            load_lights(scene_graph->light_configs);
            preprocess_meshes();
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

            ret += _size_in_bytes(_cpu_lights);

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
            _cpu_lights.clear();
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
            _cpu_lights.shrink_to_fit();
        }

        std::string Scene::description() const {
            float size_in_MB = size_in_bytes() / sqr(1024);

            return string_printf("scene data occupy %f MB, instance triangle is %u,"
                                 " instance vertices is %u, light num is %u",
                                 size_in_MB,
                                 _inst_triangle_num,
                                 _inst_vertices_num,
                                 _cpu_lights.size());
        }
    }
}