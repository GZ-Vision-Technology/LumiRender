//
// Created by Zero on 2021/3/24.
//

#include "scene.h"
#include "util/stats.h"
#include "render/materials/common.h"
#include "render/lights/common.h"
#include "render/light_samplers/common.h"
#include "render/sensors/sensor.h"

namespace luminous {
    inline namespace render {

        void Scene::reserve_geometry(const SP<SceneGraph> &scene_graph) {
            _positions.reserve(scene_graph->position_num);
            _meshes.reserve(scene_graph->mesh_num);
            _tex_coords.reserve(scene_graph->tex_coords_num);
            _normals.reserve(scene_graph->normal_num);
            _triangles.reserve(scene_graph->tri_num);

            _transforms.reserve(scene_graph->model_list.size());
            _inst_to_mesh_idx.reserve(scene_graph->instance_num);
            _inst_to_transform_idx.reserve(scene_graph->instance_num);
        }

        void Scene::append_light_material(vector<MaterialConfig> &material_configs) {
            TextureConfig tc;
            tc.set_full_type("ConstantTexture");
            tc.val = make_float4(0.f);
            _tex_configs.push_back(tc);

            MaterialConfig mc;
            mc.set_full_type("MatteMaterial");
            mc.diffuse_tex.tex_idx = _tex_configs.size() - 1;
            material_configs.push_back(mc);
        }

        void Scene::relevance_light_and_texture(vector<LightConfig> &light_configs) {
            for (auto &config : light_configs) {
                config.fill_tex_config(_tex_configs);
                if (config.type() == full_type("Envmap")) {
                    const Image &image = _images[config.texture_config.image_idx];
                    std::vector<float> vec = Envmap::create_distribution(image);
                    config.distribution_idx = _distribution_mgr.distribution2ds.size();
                    _distribution_mgr.add_distribution2d(vec, image.width(), image.height());
                }
            }
        }

        void Scene::load_lights(const vector<LightConfig> &light_configs, const LightSamplerConfig &lsc) {
            _lights.init(light_configs.size());
            for (const auto &lc : light_configs) {
                lc.scene_box = _scene_box;
                if (lc.type() == type_name<Envmap>()) {
                    ++_infinite_light_num;
                }
                _lights.add_element(lc);
            }

            // put the infinite light to first
            std::sort(_lights.begin(), _lights.end(), [](const Light &v1, const Light &v2) {
                return v1.is_infinite() > v2.is_infinite();
            });

            _light_sampler.init(1);
            _light_sampler.add_element(lsc);
        }

        void Scene::preprocess_meshes() {
            vector<Distribution1DBuilder> builders;
            auto process_mesh = [&](MeshHandle mesh) {
                if (!mesh.has_distribute()) {
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
                _distribution_mgr.add_distribution(builder, true);
            }
        }

        void Scene::relevance_material_and_texture(vector<MaterialConfig> &material_configs) {
            for (auto &mat_config : material_configs) {
                mat_config.fill_tex_configs(_tex_configs);
            }
        }

        void Scene::convert_geometry_data(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("convert scene data")
            index_t vert_offset = 0u;
            index_t tri_offset = 0u;
            vector<TextureConfig> tex_configs;
            index_t material_count = scene_graph->material_configs.size();
            reserve_geometry(scene_graph);
            for (const SP<const Model> &model : scene_graph->model_list) {
                int64_t model_mat_idx = lstd::find_index_if(scene_graph->material_configs,
                                                            [&](const MaterialConfig &val) {
                                                                return model->has_custom_material() &&
                                                                       val.name == model->custom_material_name;
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
                lstd::append(scene_graph->material_configs, model->materials);
                lstd::append(tex_configs, model->textures);
            }

            lstd::append(_tex_configs, scene_graph->tex_configs);
            relevance_material_and_texture(scene_graph->material_configs);
            append_light_material(scene_graph->material_configs);
            init_materials(scene_graph);

            auto compute_surface_area = [&](MeshHandle mesh, const Transform &transform) {
                auto tri_start = mesh.triangle_offset;
                auto tri_end = tri_start + mesh.triangle_count;
                float surface_area = 0;
                BufferView<const float3> mesh_positions = _positions.const_host_buffer_view(mesh.vertex_offset,
                                                                                            mesh.vertex_count);
                for (size_t i = tri_start; i < tri_end; ++i) {
                    TriangleHandle tri = _triangles[i];
                    float3 p0 = transform.apply_point(mesh_positions[tri.i]);
                    float3 p1 = transform.apply_point(mesh_positions[tri.j]);
                    float3 p2 = transform.apply_point(mesh_positions[tri.k]);
                    surface_area += triangle_area(p0, p1, p2);
                }
                return surface_area;
            };

            index_t distribute_idx = 0;
            for (const SP<const ModelInstance> &instance : scene_graph->instance_list) {
                const SP<const Model> &model = scene_graph->model_list[instance->model_idx];
                for (const SP<const Mesh> &mesh : model->meshes) {
                    _inst_to_transform_idx.push_back(_transforms.size());
                    _scene_box.extend(instance->o2w.apply_box(mesh->aabb));
                    if (nonzero(instance->emission)) {
                        LightConfig lc;
                        lc.emission = instance->emission;
                        lc.set_full_type("AreaLight");
                        lc.instance_idx = _inst_to_mesh_idx.size();
                        MeshHandle &mesh_handle = _meshes[mesh->idx_in_meshes];
                        mesh_handle.light_idx = scene_graph->light_configs.size();
                        lc.surface_area = compute_surface_area(mesh_handle, instance->o2w);
                        scene_graph->light_configs.push_back(lc);
                        mesh_handle.material_idx = _materials.size() - 1;
                        if (!mesh_handle.has_distribute()) {
                            mesh_handle.distribute_idx = distribute_idx++;
                        }
                    }
                    _inst_to_mesh_idx.push_back(mesh->idx_in_meshes);
                    _inst_triangle_num += mesh->triangles.size();
                    _inst_vertices_num += mesh->positions.size();
                }
                _transforms.push_back(instance->o2w);
            }
        }

        void Scene::init_lights(const SP<SceneGraph> &scene_graph) {
            preprocess_meshes();
            relevance_light_and_texture(scene_graph->light_configs);
            load_lights(scene_graph->light_configs, scene_graph->light_sampler_config);
        }

        void Scene::preload_textures(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("preload_textures")
            _textures.reserve(_tex_configs.size());
            for (auto &tc : _tex_configs) {
                if (tc.type() == type_name<ImageTexture>() && !tc.fn.empty()) {
                    if (luminous_fs::path(tc.fn).is_relative()) {
                        auto path = _context->scene_path() / tc.fn;
                        tc.fn = path.string();
                    }
                    Image image = Image::load(tc.fn, tc.color_space);
                    DTexture &texture = _device->allocate_texture(image.pixel_format(), image.resolution());
                    texture.copy_from(image);
                    tc.handle = texture.tex_handle();
                    tc.pixel_format = texture.pixel_format();
                    _texture_num += 1;
                    _texture_size_in_byte += image.size_in_bytes();
                    tc.image_idx = _images.size();
                    _images.push_back(move(image));
                }
                _textures.push_back(Texture::create(tc));
            }
        }

        void Scene::init_materials(const SP<SceneGraph> &scene_graph) {
            _materials.init(scene_graph->material_configs.size());
            for (const auto &mat_config : scene_graph->material_configs) {
                _materials.add_element(mat_config);
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

//            ret += _lights.size_in_bytes();
            ret += _distribution_mgr.size_in_bytes();
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
                _distribution_mgr.clear();
            }
            {
                _materials.clear();
                _textures.clear();
            }
            {
                _tex_configs.clear();
                _scene_box = Box3f();
            }
            _accelerator->clear();
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
                _distribution_mgr.shrink_to_fit();
            }
            {
                _materials.shrink_to_fit();
                _textures.shrink_to_fit();
            }
            {
                _tex_configs.shrink_to_fit();
            }
        }

        std::string Scene::description() const {
            float size_in_MB = size_in_bytes() / sqr(1024.f);

            return string_printf("all scene data occupy %.5f MB,\n"
                                 "texture num is %u, texture size is %.5f MB,\n"
                                 "instance triangle is %u,\n"
                                 "instance vertices is %u,\n"
                                 "scene box is %s, \n"
                                 "light num is %u \n",
                                 size_in_MB,
                                 _texture_num,
                                 _texture_size_in_byte / sqr(1024.f),
                                 _inst_triangle_num,
                                 _inst_vertices_num,
                                 _scene_box.to_string().c_str(),
                                 _lights.size());
        }

    }
}