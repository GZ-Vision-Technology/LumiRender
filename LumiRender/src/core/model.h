//
// Created by Zero on 2021/2/16.
//


#pragma once

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "graphics/string_util.h"
#include "render/include/shape.h"

using namespace std;

namespace luminous {
    inline namespace utility {

        class Model;

        class ModelCache {
        private:
            static ModelCache *s_model_cache;
            map <string, shared_ptr<const Model>> _model_map;

            [[nodiscard]] shared_ptr<const Model> load_model(const std::string &path,
                                                             uint32_t subdiv_level);

            [[nodiscard]] bool inline is_contain(const std::string &key) const {
                return _model_map.find(key) != _model_map.end();
            }

            [[nodiscard]] static string cal_key(const string &path, uint32_t subdiv_level) {
                return string_printf("%s_subdiv_%u", path.c_str(), subdiv_level);
            }

        public:

            [[nodiscard]] const shared_ptr<const Model> &get_model(const std::string &path,
                                                                   uint32_t subdiv_level);

            [[nodiscard]] static ModelCache *instance();

        };
        using std::vector;
        struct Model {
            Model(vector<shared_ptr<const Mesh>> meshes):meshes(std::move(meshes)) {}
            vector<shared_ptr<const Mesh>> meshes;
        };

    } // luminous::utility
} // luminous