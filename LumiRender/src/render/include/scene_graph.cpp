//
// Created by Zero on 2021/2/21.
//

#include "scene_graph.h"
#include "util/stats.h"

namespace luminous {
    inline namespace render {


        const static std::vector<luminous::float3> g_vertices =
                {
                        // Floor  -- white lambert
                        luminous::float3(    0.0f,    0.0f,    0.0f),
                        luminous::float3(    0.0f,    0.0f,  559.2f),
                        luminous::float3(  556.0f,    0.0f,  559.2f),
                        luminous::float3(    0.0f,    0.0f,    0.0f),
                        luminous::float3(  556.0f,    0.0f,  559.2f),
                        luminous::float3(  556.0f,    0.0f,    0.0f),

                        // Ceiling -- white lambert
                        luminous::float3(    0.0f,  548.8f,    0.0f),
                        luminous::float3(  556.0f,  548.8f,    0.0f),
                        luminous::float3(  556.0f,  548.8f,  559.2f),

                        luminous::float3(    0.0f,  548.8f,    0.0f),
                        luminous::float3(  556.0f,  548.8f,  559.2f),
                        luminous::float3(    0.0f,  548.8f,  559.2f),

                        // Back wall -- white lambert
                        luminous::float3(    0.0f,    0.0f,  559.2f),
                        luminous::float3(    0.0f,  548.8f,  559.2f),
                        luminous::float3(  556.0f,  548.8f,  559.2f),

                        luminous::float3(    0.0f,    0.0f,  559.2f),
                        luminous::float3(  556.0f,  548.8f,  559.2f),
                        luminous::float3(  556.0f,    0.0f,  559.2f),

                        // Right wall -- green lambert
                        luminous::float3(    0.0f,    0.0f,    0.0f),
                        luminous::float3(    0.0f,  548.8f,    0.0f),
                        luminous::float3(    0.0f,  548.8f,  559.2f),

                        luminous::float3(    0.0f,    0.0f,    0.0f),
                        luminous::float3(    0.0f,  548.8f,  559.2f),
                        luminous::float3(    0.0f,    0.0f,  559.2f),

                        // Left wall -- red lambert
                        luminous::float3(  556.0f,    0.0f,    0.0f),
                        luminous::float3(  556.0f,    0.0f,  559.2f),
                        luminous::float3(  556.0f,  548.8f,  559.2f),

                        luminous::float3(  556.0f,    0.0f,    0.0f),
                        luminous::float3(  556.0f,  548.8f,  559.2f),
                        luminous::float3(  556.0f,  548.8f,    0.0f),

                        // Short block -- white lambert
                        luminous::float3(  130.0f,  165.0f,   65.0f),
                        luminous::float3(   82.0f,  165.0f,  225.0f),
                        luminous::float3(  242.0f,  165.0f,  274.0f),

                        luminous::float3(  130.0f,  165.0f,   65.0f),
                        luminous::float3(  242.0f,  165.0f,  274.0f),
                        luminous::float3(  290.0f,  165.0f,  114.0f),

                        luminous::float3(  290.0f,    0.0f,  114.0f),
                        luminous::float3(  290.0f,  165.0f,  114.0f),
                        luminous::float3(  240.0f,  165.0f,  272.0f),

                        luminous::float3(  290.0f,    0.0f,  114.0f),
                        luminous::float3(  240.0f,  165.0f,  272.0f),
                        luminous::float3(  240.0f,    0.0f,  272.0f),

                        luminous::float3(  130.0f,    0.0f,   65.0f),
                        luminous::float3(  130.0f,  165.0f,   65.0f),
                        luminous::float3(  290.0f,  165.0f,  114.0f),

                        luminous::float3(  130.0f,    0.0f,   65.0f),
                        luminous::float3(  290.0f,  165.0f,  114.0f),
                        luminous::float3(  290.0f,    0.0f,  114.0f),

                        luminous::float3(   82.0f,    0.0f,  225.0f),
                        luminous::float3(   82.0f,  165.0f,  225.0f),
                        luminous::float3(  130.0f,  165.0f,   65.0f),

                        luminous::float3(   82.0f,    0.0f,  225.0f),
                        luminous::float3(  130.0f,  165.0f,   65.0f),
                        luminous::float3(  130.0f,    0.0f,   65.0f),

                        luminous::float3(  240.0f,    0.0f,  272.0f),
                        luminous::float3(  240.0f,  165.0f,  272.0f),
                        luminous::float3(   82.0f,  165.0f,  225.0f),

                        luminous::float3(  240.0f,    0.0f,  272.0f),
                        luminous::float3(   82.0f,  165.0f,  225.0f),
                        luminous::float3(   82.0f,    0.0f,  225.0f),

                        // Tall block -- white lambert
                        luminous::float3(  423.0f,  330.0f,  247.0f),
                        luminous::float3(  265.0f,  330.0f,  296.0f),
                        luminous::float3(  314.0f,  330.0f,  455.0f),

                        luminous::float3(  423.0f,  330.0f,  247.0f),
                        luminous::float3(  314.0f,  330.0f,  455.0f),
                        luminous::float3(  472.0f,  330.0f,  406.0f),

                        luminous::float3(  423.0f,    0.0f,  247.0f),
                        luminous::float3(  423.0f,  330.0f,  247.0f),
                        luminous::float3(  472.0f,  330.0f,  406.0f),

                        luminous::float3(  423.0f,    0.0f,  247.0f),
                        luminous::float3(  472.0f,  330.0f,  406.0f),
                        luminous::float3(  472.0f,    0.0f,  406.0f),

                        luminous::float3(  472.0f,    0.0f,  406.0f),
                        luminous::float3(  472.0f,  330.0f,  406.0f),
                        luminous::float3(  314.0f,  330.0f,  456.0f),

                        luminous::float3(  472.0f,    0.0f,  406.0f),
                        luminous::float3(  314.0f,  330.0f,  456.0f),
                        luminous::float3(  314.0f,    0.0f,  456.0f),

                        luminous::float3(  314.0f,    0.0f,  456.0f),
                        luminous::float3(  314.0f,  330.0f,  456.0f),
                        luminous::float3(  265.0f,  330.0f,  296.0f),

                        luminous::float3(  314.0f,    0.0f,  456.0f),
                        luminous::float3(  265.0f,  330.0f,  296.0f),
                        luminous::float3(  265.0f,    0.0f,  296.0f),

                        luminous::float3(  265.0f,    0.0f,  296.0f),
                        luminous::float3(  265.0f,  330.0f,  296.0f),
                        luminous::float3(  423.0f,  330.0f,  247.0f),

                        luminous::float3(  265.0f,    0.0f,  296.0f),
                        luminous::float3(  423.0f,  330.0f,  247.0f),
                        luminous::float3(  423.0f,    0.0f,  247.0f),

                        // Ceiling light -- emmissive
                        luminous::float3(  343.0f,  548.6f,  227.0f),
                        luminous::float3(  213.0f,  548.6f,  227.0f),
                        luminous::float3(  213.0f,  548.6f,  332.0f),

                        luminous::float3(  343.0f,  548.6f,  227.0f),
                        luminous::float3(  213.0f,  548.6f,  332.0f),
                        luminous::float3(  343.0f,  548.6f,  332.0f)
                };

        std::vector<luminous::TriangleHandle> tri_list;


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

        SP<Model> SceneGraph::create_shape(const ShapeConfig &config) {
            if (config.type() == "model") {
                config.fn = (_context->scene_path() / config.fn).string();
                auto model = std::make_shared<Model>(config);
                return model;
            } else if (config.type() == "quad") {
                auto model = std::make_shared<Model>();
                auto x = config.width / 2;
                auto y = config.height / 2;
                Box3f aabb;
                vector<float3> P{make_float3(x, y, 0),
                                 make_float3(x, -y, 0),
                                 make_float3(-x, y, 0),
                                 make_float3(-x, -y, 0)};

                for (auto p : P) {
                    aabb.extend(p);
                }
                
                vector<float3> N(4, make_float3(0, 0, -1));

                vector<float2> UV{make_float2(1, 1),
                                  make_float2(1, 0),
                                  make_float2(0, 1),
                                  make_float2(0, 0)};

                vector<TriangleHandle> triangles{TriangleHandle{0,1,2},
                                                 TriangleHandle{2,1,3}};

                auto mesh = std::make_shared<Mesh>(move(P),move(N), move(UV), move(triangles), aabb);
                model->meshes.push_back(mesh);
                model->custom_material_name = config.material_name;
                return model;
            } else {
                LUMINOUS_ERROR("unknown shape type !")
            }
        }

        void SceneGraph::create_shape_instance(const ShapeConfig &config) {
            auto key = gen_key(config);
            if (!is_contain(key)) {
                auto mp = create_shape(config);
                mp->key = key;
                _key_to_idx[key] = model_list.size();
                model_list.push_back(mp);
            }
            uint idx = _key_to_idx[key];
            Transform o2w = config.o2w.create();
            auto instance = std::make_shared<ModelInstance>(idx, o2w, config.name.c_str(), config.emission);
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