//
// Created by Zero on 2021/3/24.
//

#include "scene.h"
#include "render/include/distribution.h"
#include "util/stats.h"

namespace luminous {
    inline namespace render {

        void Scene::load_lights(const vector <LightConfig> &light_configs, const LightSamplerConfig &lsc) {
            _lights.reserve(light_configs.size());
            for (const auto &lc : light_configs) {
                _lights.push_back(Light::create(lc));
            }
            auto light_sampler = LightSampler::create(lsc);
            _light_sampler.reset(&light_sampler);
        }

        void Scene::preprocess_meshes() {
            vector <Distribution1DBuilder> builders;
            auto process_mesh = [&](MeshHandle mesh) {
                if (mesh.distribute_idx == -1) {
                    return;
                }
                uint start = mesh.triangle_offset;
                uint end = start + mesh.triangle_count;
                vector<float> areas;
                areas.reserve(mesh.triangle_count);
                const float3 *pos = &_positions[mesh.vertex_offset];
                for (int i = start; i < end; ++i) {
                    TriangleHandle tri = _triangles[i];
                    float3 p0 = pos[tri.i];
                    float3 p1 = pos[tri.j];
                    float3 p2 = pos[tri.k];
                    float area = triangle_area(p0, p1, p2);
                    areas.push_back(area);
                }
                auto builder = Distribution1D::create_builder(move(areas));
                builders.push_back(builder);
            };

            for (const auto &mesh : _meshes) {
                process_mesh(mesh);
            }
            for (const auto &builder : builders) {
                _emission_distrib.add_distribute(builder);
            }
        }

        bool is_contain(const vector <TextureConfig> &tex_configs, const TextureConfig &texture_config) {
            return std::any_of(tex_configs.cbegin(), tex_configs.cend(), [&](const auto &config) {
                return config == texture_config;
            });
        }

        void Scene::init_texture_configs(const vector <MaterialConfig> &material_configs) {
            for (const auto &mat_config : material_configs) {
                auto tex_configs = mat_config.tex_configs();
                for (const auto &tex_config : tex_configs) {
                    if (is_contain(_tex_configs, tex_config)) {
                        continue;
                    }
                    _tex_configs.push_back(tex_config);
                }
            }
        }

        void Scene::convert_data(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("convert scene data start!")
            index_t vert_offset = 0u;
            index_t tri_offset = 0u;

            vector <TextureConfig> tex_configs;
            index_t material_count = 0u;
            for (const SP<const Model> &model : scene_graph->model_list) {
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _positions.append(mesh->positions);
                    _normals.append(mesh->normals);
                    _tex_coords.append(mesh->tex_coords);
                    _triangles.append(mesh->triangles);

                    index_t vert_count = mesh->positions.size();
                    index_t tri_count = mesh->triangles.size();
                    mesh->idx_in_meshes = _meshes.size();
                    _meshes.emplace_back(vert_offset, tri_offset, vert_count, tri_count,
                                         mesh->mat_idx + material_count);
                    vert_offset += vert_count;
                    tri_offset += tri_count;
                }
                material_count += model->materials.size();
                append(_material_configs, model->materials);
                append(tex_configs, model->textures);
            }

            init_texture_configs(_material_configs);

            index_t distribute_idx = 0;
            for (const SP<const ModelInstance> &instance : scene_graph->instance_list) {
                const SP<const Model> &model = scene_graph->model_list[instance->model_idx];
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _inst_to_transform_idx.push_back(_transforms.size());

                    if (!instance->emission.is_zero()) {
                        LightConfig lc;
                        lc.emission = instance->emission;
                        lc.set_full_type("AreaLight");
                        lc.instance_idx = _inst_to_mesh_idx.size();
                        scene_graph->light_configs.push_back(lc);
                        MeshHandle &mesh_handle = _meshes[mesh->idx_in_meshes];
                        if (mesh_handle.distribute_idx == index_t(-1)) {
                            mesh_handle.distribute_idx = distribute_idx++;
                        }
                    }
                    _inst_to_mesh_idx.push_back(mesh->idx_in_meshes);

                    _inst_triangle_num += mesh->triangles.size();
                    _inst_vertices_num += mesh->positions.size();
                }
                _transforms.push_back(instance->o2w.mat4x4());
            }
            load_lights(scene_graph->light_configs, scene_graph->light_sampler_config);
            preprocess_meshes();
            shrink_to_fit();
        }

        size_t Scene::size_in_bytes() const {
            size_t ret = _triangles.size_in_bytes();
            ret += _tex_coords.size_in_bytes();
            ret += _positions.size_in_bytes();
            ret += _normals.size_in_bytes();

            ret += _meshes.size_in_bytes();
            ret += _transforms.size_in_bytes();
            ret += _inst_to_mesh_idx.size_in_bytes();
            ret += _inst_to_transform_idx.size_in_bytes();

            ret += _lights.size_in_bytes();
            ret += _emission_distrib.size_in_bytes();

            return ret;
        }

        void Scene::clear() {
            {
                _triangles.clear();
                _tex_coords.clear();
                _positions.clear();
                _normals.clear();
                _meshes.clear();
            }
            {
                _transforms.clear();
                _inst_to_mesh_idx.clear();
                _inst_to_transform_idx.clear();
            }
            {
                _lights.clear();
                _emission_distrib.clear();
            }
        }

        void Scene::shrink_to_fit() {
            {
                _triangles.shrink_to_fit();
                _tex_coords.shrink_to_fit();
                _positions.shrink_to_fit();
                _normals.shrink_to_fit();
                _meshes.shrink_to_fit();
            }
            {
                _transforms.shrink_to_fit();
                _inst_to_mesh_idx.shrink_to_fit();
                _inst_to_transform_idx.shrink_to_fit();
            }
            {
                _lights.shrink_to_fit();
                _emission_distrib.shrink_to_fit();
            }
        }

        std::string Scene::description() const {
            float size_in_MB = size_in_bytes() / sqr(1024.f);

            return string_printf("scene data occupy %f MB, instance triangle is %u,"
                                 " instance vertices is %u, light num is %u",
                                 size_in_MB,
                                 _inst_triangle_num,
                                 _inst_vertices_num,
                                 _lights.size());
        }

    }
}