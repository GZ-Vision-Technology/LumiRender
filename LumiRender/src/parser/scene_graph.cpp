//
// Created by Zero on 2021/2/21.
//

#include "scene_graph.h"
#include "util/stats.h"

namespace luminous {
    inline namespace render {

        string gen_key(const string &fn, uint subdiv_level = 0) {
            return string_printf("%s_subdiv_%u", fn.c_str(), subdiv_level);
        }

        string gen_key(const ShapeConfig sc) {
            if (sc.type() == "model") {
                return string_printf("%s_subdiv_%u", sc.fn.c_str(), sc.subdiv_level);
            } else {
                return sc.name;
            }
        }

        Model SceneGraph::create_quad_y(const ShapeConfig &config) {
            auto model = Model();
            auto width = config.width / 2;
            auto height = config.height / 2;
            Box3f aabb;
            vector<float3> P{make_float3(width, 0, height),
                             make_float3(width, 0, -height),
                             make_float3(-width, 0, height),
                             make_float3(-width, 0, -height)};

            for (auto p : P) {
                aabb.extend(p);
            }

            vector<float3> N(4, make_float3(0, 0, 0));

            vector<float2> UV{make_float2(1, 1),
                              make_float2(1, 0),
                              make_float2(0, 1),
                              make_float2(0, 0)};

            vector<TriangleHandle> triangles{TriangleHandle{1, 2, 0},
                                             TriangleHandle{1, 3, 2}};

            auto mesh = Mesh(move(P), move(N), move(UV), move(triangles), aabb);
            model.meshes.push_back(mesh);
            model.custom_material_name = config.material_name;
            update_counter(model);
            return model;
        }

        Model SceneGraph::create_quad(const ShapeConfig &config) {
            auto model = Model();
            auto width = config.width / 2;
            auto height = config.height / 2;
            Box3f aabb;
            vector<float3> P{make_float3(width, height, 0),
                             make_float3(width, -height, 0),
                             make_float3(-width, height, 0),
                             make_float3(-width, -height, 0)};

            for (auto p : P) {
                aabb.extend(p);
            }

            vector<float3> N(4, make_float3(0, 0, 0));

            vector<float2> UV{make_float2(1, 1),
                              make_float2(1, 0),
                              make_float2(0, 1),
                              make_float2(0, 0)};

            vector<TriangleHandle> triangles{TriangleHandle{1, 0, 2},
                                             TriangleHandle{1, 2, 3}};

            auto mesh = Mesh(move(P), move(N), move(UV), move(triangles), aabb);
            model.meshes.push_back(mesh);
            model.custom_material_name = config.material_name;
            update_counter(model);
            return model;
        }

        Model SceneGraph::create_cube(const ShapeConfig &config) {
            float x = config.x;
            float y = config.y;
            float z = config.z;
            y = y == 0 ? x : y;
            z = z == 0 ? y : z;
            x = x / 2.f;
            y = y / 2.f;
            z = z / 2.f;
            auto P = vector<float3>{
                    float3(-x, -y, z), float3(x, -y, z), float3(-x, y, z), float3(x, y, z), // +z
                    float3(-x, y, -z), float3(x, y, -z), float3(-x, -y, -z), float3(x, -y, -z), // -z
                    float3(-x, y, z), float3(x, y, z), float3(-x, y, -z), float3(x, y, -z),  // +y
                    float3(-x, -y, z), float3(x, -y, z), float3(-x, -y, -z), float3(x, -y, -z), // -y
                    float3(x, -y, z), float3(x, y, z), float3(x, y, -z), float3(x, -y, -z), // +x
                    float3(-x, -y, z), float3(-x, y, z), float3(-x, y, -z), float3(-x, -y, -z), // -x
            };
            auto N = vector<float3>{
                    float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1),
                    float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1),
                    float3(0, 1, 0), float3(0, 1, 0), float3(0, 1, 0), float3(0, 1, 0),
                    float3(0, -1, 0), float3(0, -1, 0), float3(0, -1, 0), float3(0, -1, 0),
                    float3(1, 0, 0), float3(1, 0, 0), float3(1, 0, 0), float3(1, 0, 0),
                    float3(-1, 0, 0), float3(-1, 0, 0), float3(-1, 0, 0), float3(-1, 0, 0),
            };
            auto UVs = vector<float2>{
                    float2(0, 0), float2(0, 0), float2(0, 0), float2(0, 0),
                    float2(0, 0), float2(0, 0), float2(0, 0), float2(0, 0),
                    float2(0, 0), float2(0, 0), float2(0, 0), float2(0, 0),
                    float2(0, 0), float2(0, 0), float2(0, 0), float2(0, 0),
                    float2(0, 0), float2(0, 0), float2(0, 0), float2(0, 0),
                    float2(0, 0), float2(0, 0), float2(0, 0), float2(0, 0),
            };
            auto triangles = vector<TriangleHandle>{
                    TriangleHandle(0, 1, 3), TriangleHandle(0, 3, 2),
                    TriangleHandle(6, 5, 7), TriangleHandle(4, 5, 6),
                    TriangleHandle(10, 9, 11), TriangleHandle(8, 9, 10),
                    TriangleHandle(13, 14, 15), TriangleHandle(13, 12, 14),
                    TriangleHandle(18, 17, 19), TriangleHandle(17, 16, 19),
                    TriangleHandle(21, 22, 23), TriangleHandle(20, 21, 23),
            };
            Model model;
            Box3f aabb;
            for (auto p : P) {
                aabb.extend(p);
            }
            auto mesh = Mesh(move(P), move(N), move(UVs), move(triangles), aabb);
            model.meshes.push_back(mesh);
            model.custom_material_name = config.material_name;
            update_counter(model);
            return model;
        }

        Model SceneGraph::create_shape(const ShapeConfig &config) {
            if (config.type() == "model") {
                config.fn = (_context->scene_path() / config.fn).string();
                auto model = Model(config);
                update_counter(model);
                return model;
            } else if (config.type() == "quad") {
                return create_quad(config);
            } else if (config.type() == "quad_y") {
                return create_quad_y(config);
            } else if (config.type() == "mesh") {
                auto model = Model();
                Box3f aabb;
                for (auto pos : config.positions) {
                    aabb.extend(pos);
                }
                Mesh mesh(move(config.positions), move(config.normals),
                          move(config.tex_coords), move(config.triangles), aabb);
                model.meshes.push_back(mesh);
                model.custom_material_name = config.material_name;
                update_counter(model);
                return model;
            } else if (config.type() == "cube") {
                return create_cube(config);
            } else {
                LUMINOUS_ERROR("unknown shape type !")
            }
        }

        void SceneGraph::create_shape_instance(const ShapeConfig &config) {
            auto key = gen_key(config);
            if (!is_contain(key)) {
                auto mp = create_shape(config);
                mp.key = key;
                _key_to_idx[key] = model_list.size();
                model_list.push_back(mp);
            }
            uint idx = _key_to_idx[key];
            Transform o2w = config.o2w.create();
            auto instance = ModelInstance(idx, o2w, config.name.c_str(), config.emission);
            instance_num += model_list[instance.model_idx].meshes.size();
            instance_list.push_back(instance);
        }

        void SceneGraph::create_shapes() {
            TASK_TAG("create shapes")
            for (const auto &shape_config : shape_configs) {
                create_shape_instance(shape_config);
            }
        }
    }
}