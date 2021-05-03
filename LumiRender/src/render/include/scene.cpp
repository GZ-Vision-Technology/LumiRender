//
// Created by Zero on 2021/3/24.
//

#include "scene.h"
#include "render/include/distribution.h"
#include "util/stats.h"
#include "graphics/lstd/lstd.h"

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

        void Scene::relevance_material_and_texture(vector <MaterialConfig> &material_configs) {
            for (auto &mat_config : material_configs) {
                mat_config.fill_tex_configs(_tex_configs);
            }
        }

        void Scene::convert_data(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("convert scene data")
            index_t vert_offset = 0u;
            index_t tri_offset = 0u;
            vector <TextureConfig> tex_configs;
            index_t material_count = scene_graph->material_configs.size();
            for (const SP<const Model> &model : scene_graph->model_list) {
                int64_t model_mat_idx = lstd::find_index_if(scene_graph->material_configs, [&](const MaterialConfig &val){
                    return val.name == model->material_name;
                });
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _positions.append(mesh->positions);
                    _normals.append(mesh->normals);
                    _tex_coords.append(mesh->tex_coords);
                    _triangles.append(mesh->triangles);

                    index_t vert_count = mesh->positions.size();
                    index_t tri_count = mesh->triangles.size();
                    mesh->idx_in_meshes = _meshes.size();

                    int mesh_mat_idx = model_mat_idx == -1 ? mesh->mat_idx + material_count : model_mat_idx;

                    CONTINUE_IF_TIPS(mesh_mat_idx == -1, "warning :mesh have no material\n")
                    _meshes.emplace_back(vert_offset, tri_offset, vert_count, tri_count,
                                         mesh_mat_idx);
                    vert_offset += vert_count;
                    tri_offset += tri_count;
                }
                material_count += model->materials.size();
                append(scene_graph->material_configs, model->materials);
                append(tex_configs, model->textures);
            }

            append(_tex_configs, scene_graph->tex_configs);
            relevance_material_and_texture(scene_graph->material_configs);
            init_materials(scene_graph);

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

        void Scene::init_materials(const SP<SceneGraph> &scene_graph) {
            for (const auto& mat_config : scene_graph->material_configs) {
                _materials.push_back(Material::create(mat_config));
            }
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
            ret += _texture_size_in_byte;

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
            {
                _materials.clear();
                _textures.clear();
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
            {
                _materials.shrink_to_fit();
                _textures.shrink_to_fit();
            }
        }

        std::string Scene::description() const {
            float size_in_MB = size_in_bytes() / sqr(1024.f);

            return string_printf("all scene data occupy %f MB,\n"
                                 "texture num is %u, texture size is %f MB,\n"
                                 "instance triangle is %u,\n"
                                 "instance vertices is %u,\n"
                                 "light num is %u",
                                 size_in_MB,
                                 _texture_num,
                                 _texture_size_in_byte / sqr(1024.f),
                                 _inst_triangle_num,
                                 _inst_vertices_num,
                                 _lights.size());
        }
    }
}