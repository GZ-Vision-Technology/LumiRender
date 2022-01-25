//
// Created by Zero on 2021/3/7.
//

#include "shape.h"
#include "parser/assimp_parser.h"
#include "core/logging.h"

namespace luminous {
    inline namespace render {

        Model::Model(const ShapeConfig &sc) {
            Assimp::Importer ai_importer;
            luminous_fs::path path = sc.fn;
            directory = path.parent_path();

            auto ai_scene = AssimpParser::load_scene(sc.fn, ai_importer, sc.swap_handed, sc.smooth, false);

            LUMINOUS_EXCEPTION_IF(
                    ai_scene == nullptr || (ai_scene->mFlags & static_cast<uint>(AI_SCENE_FLAGS_INCOMPLETE)) ||
                    ai_scene->mRootNode == nullptr,
                    "Failed to load triangle mesh: ", ai_importer.GetErrorString());

            meshes = AssimpParser::parse_meshes(ai_scene, sc.subdiv_level);

            custom_material_name = sc.material_name;
            if (custom_material_name.empty()) {
                materials = AssimpParser::parse_materials(ai_scene, directory, sc.use_normal_map);
            }
        }
    } // luminous::render
} // luminous