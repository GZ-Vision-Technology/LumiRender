//
// Created by Zero on 2021/2/21.
//

#include "scene_graph.h"


namespace luminous {
    inline namespace render {

        string gen_key(const string &fn, uint subdiv_level = 0) {
            return string_printf("%s_subdiv_%u", fn.c_str(), subdiv_level);
        }

        SP<Model> SceneGraph::create_shape(const ShapeConfig &config) {
            if (config.type == "model") {
                auto path = _context->scene_path() / config.fn;
                return make_shared<Model>(path, config.subdiv_level);
            } else if (config.type == "quad") {
                auto model = make_shared<Model>();
                auto x = config.width / 2;
                auto y = config.height / 2;

                vector<float3> P{make_float3(x, y, 0),
                                 make_float3(x, -y, 0),
                                 make_float3(-x, y, 0),
                                 make_float3(-x, -y, 0)};

                vector<float3> N(4, make_float3(0, 0, 1));

                vector<float2> UV{make_float2(1, 1),
                                  make_float2(1, 0),
                                  make_float2(0, 1),
                                  make_float2(0, 0)};

                vector<TriangleHandle> triangles{TriangleHandle{0,1,2},
                                                 TriangleHandle{3,1,2}};

                auto mesh = make_shared<Mesh>(move(P),move(N), move(UV), move(triangles));
                model->meshes.push_back(mesh);

                return model;
            }
        }

        void SceneGraph::create_shape_instance(const ShapeConfig &config) {
            auto key = gen_key(config.fn, config.subdiv_level);
            if (!is_contain(key)) {
                auto mp = create_shape(config);
                mp->key = key;
                _key_to_idx[key] = model_list.size();
                model_list.push_back(mp);
            }
            uint idx = _key_to_idx[key];
            Transform o2w = config.o2w.create();
            auto instance = make_shared<ModelInstance>(idx, o2w, config.name.c_str());
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