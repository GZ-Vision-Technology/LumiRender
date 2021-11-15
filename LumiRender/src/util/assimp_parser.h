//
// Created by Zero on 15/11/2021.
//


#pragma once

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/Subdivision.h>
#include <assimp/scene.h>
#include "parser.h"

namespace luminous {
    inline namespace utility {
        class AssimpParser : public Parser {
        private:
            Assimp::Importer _ai_importer;
            const aiScene *_ai_scene{nullptr};
            luminous_fs::path directory;
        public:

            LM_NODISCARD static const aiScene *load_scene(const luminous_fs::path &fn,
                                                          Assimp::Importer &ai_importer,
                                                          bool swap_handed = false,
                                                          bool smooth = true);

            LM_NODISCARD static std::vector<Mesh> parse_meshes(const aiScene *ai_scene,
                                                               uint32_t subdiv_level = 0u);

            void load(const luminous_fs::path &fn) override;

            LM_NODISCARD SP<SceneGraph> parse() const override;
        };
    }
}