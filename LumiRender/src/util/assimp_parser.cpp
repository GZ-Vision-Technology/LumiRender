//
// Created by Zero on 15/11/2021.
//


#include "assimp_parser.h"

namespace luminous {
    inline namespace utility {

        void AssimpParser::load(const luminous_fs::path &fn) {

        }

        SP<SceneGraph> AssimpParser::parse() const {
            return luminous::SP<SceneGraph>();
        }
    }
}