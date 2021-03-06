//
// Created by Zero on 2021/2/21.
//

#include "scene_graph.h"


namespace luminous {
    inline namespace render {

        string gen_key(const string &fn, uint subdiv_level = 0) {
            return string_printf("%s_subdiv_%u", fn.c_str(), subdiv_level);
        }

        SP<const Model> SceneGraph::create_model_instance(const string &fn, uint subdiv_level) {
            auto key = gen_key(fn, subdiv_level);
            if (is_contain(key)) {

            }
            return nullptr;
        }

        void SceneGraph::create_scene() {
            for (const auto& shape_config : shape_configs) {
                auto path = _context->scene_path() / shape_config.fn;
                cout << path << endl;
                Model m(path);
            }
        }
    }
}