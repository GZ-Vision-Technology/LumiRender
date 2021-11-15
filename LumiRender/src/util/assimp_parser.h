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
            const aiScene* _ai_scene{nullptr};
        public:
            void load(const luminous_fs::path &fn) override;

            LM_NODISCARD SP<SceneGraph> parse() const override;
        };
    }
}