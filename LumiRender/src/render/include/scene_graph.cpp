//
// Created by Zero on 2021/2/21.
//

#include "scene_graph.h"


namespace luminous {
    inline namespace render {

        string gen_key(const string &fn, uint subdiv_level = 0) {
            return string_printf("%s_subdiv_%u", fn.c_str(), subdiv_level);
        }

        void SceneGraph::create_shape_instance(const ShapeConfig &config) {
            auto key = gen_key(config.fn, config.subdiv_level);
            if (!is_contain(key)) {
                auto path = _context->scene_path() / config.fn;
                auto mp = make_shared<Model>(path, config.subdiv_level);
                mp->key = key;
                model_list.push_back(mp);
                _key_to_idx[key] = model_list.size();
            }
            uint idx = _key_to_idx[key];
            Transform o2w = config.o2w.create();
            auto instance = make_shared<ModelInstance>(idx, o2w, config.name.c_str());
            instance_list.push_back(instance);
        }

        void SceneGraph::create_shapes() {
            TASK_TAG("create shapes")
            for (const auto& shape_config : shape_configs) {
                create_shape_instance(shape_config);
            }
        }
    }
}