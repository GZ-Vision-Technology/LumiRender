//
// Created by Zero on 15/11/2021.
//


#include "assimp_parser.h"

namespace luminous {
    inline namespace utility {

        const aiScene *AssimpParser::load_scene(const luminous_fs::path &fn, Assimp::Importer &ai_importer,
                                                bool swap_handed, bool smooth) {
            ai_importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                           aiComponent_COLORS |
                                           aiComponent_BONEWEIGHTS |
                                           aiComponent_ANIMATIONS |
                                           aiComponent_LIGHTS |
                                           aiComponent_CAMERAS |
                                           aiComponent_TEXTURES |
                                           aiComponent_MATERIALS);
            LUMINOUS_INFO("Loading triangle mesh: ", fn);
            aiPostProcessSteps normal_flag = smooth ? aiProcess_GenSmoothNormals : aiProcess_GenNormals;
            auto post_process_steps = aiProcess_JoinIdenticalVertices |
                                      normal_flag |
                                      aiProcess_PreTransformVertices |
                                      aiProcess_ImproveCacheLocality |
                                      aiProcess_FixInfacingNormals |
                                      aiProcess_FindInvalidData |
                                      aiProcess_GenUVCoords |
                                      aiProcess_TransformUVCoords |
                                      aiProcess_OptimizeMeshes |
                                      aiProcess_FlipUVs;
            post_process_steps = swap_handed ?
                                 post_process_steps | aiProcess_ConvertToLeftHanded :
                                 post_process_steps;
            auto ai_scene = ai_importer.ReadFile(fn.string().c_str(),
                                                 post_process_steps);

            return ai_scene;
        }

        void AssimpParser::load(const luminous_fs::path &fn) {
            directory = fn.parent_path();
            _ai_scene = load_scene(fn, _ai_importer);
            LUMINOUS_EXCEPTION_IF(
                    _ai_scene == nullptr || (_ai_scene->mFlags & static_cast<uint>(AI_SCENE_FLAGS_INCOMPLETE)) ||
                    _ai_scene->mRootNode == nullptr,
                    "Failed to load scene: ", _ai_importer.GetErrorString());
        }

        SP<SceneGraph> AssimpParser::parse() const {
            return {};
        }

    }
}