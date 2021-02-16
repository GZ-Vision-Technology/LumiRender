//
// Created by Zero on 2021/2/16.
//


#pragma once

#include <map>
#include <vector>

namespace luminous {
    inline namespace utility {

        class MeshesCache {
        private:
            static MeshesCache * s_meshes_cache;

            map<string, std::vector<shared_ptr<const Mesh>>> _meshes_map;

            [[nodiscard]] static std::vector<shared_ptr<const Mesh>> load_meshes(const std::string &path,
                                                                                 uint subdiv_level);

            [[nodiscard]] bool inline is_contain(const std::string &key) const {
                return _meshes_map.find(key) != _meshes_map.end();
            }

            [[nodiscard]] static string cal_key(const string &path, uint subdiv_level) {
                return string_printf("%s_subdiv_%u", path.c_str(), subdiv_level);
            }

        public:

            [[nodiscard]] static const std::vector<shared_ptr<const Mesh>>& get_meshes(const std::string &path,
                                                                                       uint subdiv_level);

            [[nodiscard]] static MeshesCache * instance();

        };

    } // luminous::utility
} // luminous