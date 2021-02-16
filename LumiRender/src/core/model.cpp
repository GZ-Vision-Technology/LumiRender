//
// Created by Zero on 2021/2/16.
//

#include "model.h"
#include "assimp/Importer.hpp"
#include <assimp/postprocess.h>
#include <assimp/Subdivision.h>
#include <assimp/scene.h>

namespace luminous {
    inline namespace utility {
        ModelCache *ModelCache::s_model_cache = nullptr;

        ModelCache *ModelCache::instance() {
            if (s_model_cache == nullptr) {
                s_model_cache = new ModelCache();
            }
            return s_model_cache;
        }

        shared_ptr<const Model> ModelCache::load_model(const std::string &path, uint32_t subdiv_level) {
            std::vector<Mesh> meshes;
            Assimp::Importer ai_importer;
            ai_importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                           aiComponent_COLORS |
                                           aiComponent_BONEWEIGHTS |
                                           aiComponent_ANIMATIONS |
                                           aiComponent_LIGHTS |
                                           aiComponent_CAMERAS |
                                           aiComponent_TEXTURES |
                                           aiComponent_MATERIALS);
            LUMINOUS_INFO("Loading triangle mesh: ", path);
            auto ai_scene = ai_importer.ReadFile(path.c_str(),
                                                 aiProcess_JoinIdenticalVertices |
                                                 aiProcess_GenNormals |
                                                 aiProcess_PreTransformVertices |
                                                 aiProcess_ImproveCacheLocality |
                                                 aiProcess_FixInfacingNormals |
                                                 aiProcess_FindInvalidData |
                                                 aiProcess_GenUVCoords |
                                                 aiProcess_TransformUVCoords |
                                                 aiProcess_OptimizeMeshes |
                                                 aiProcess_FlipUVs);

            LUMINOUS_EXCEPTION_IF(ai_scene == nullptr || (ai_scene->mFlags & static_cast<uint>(AI_SCENE_FLAGS_INCOMPLETE)) || ai_scene->mRootNode == nullptr,
                                  "Failed to load triangle mesh: ", ai_importer.GetErrorString());

            std::vector<aiMesh *> ai_meshes(ai_scene->mNumMeshes);
            if (subdiv_level != 0u) {
                auto subdiv = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
                subdiv->Subdivide(ai_scene->mMeshes, ai_scene->mNumMeshes, ai_meshes.data(), subdiv_level);
            } else {
                std::copy(ai_scene->mMeshes, ai_scene->mMeshes + ai_scene->mNumMeshes, ai_meshes.begin());
            }

            meshes.reserve(ai_meshes.size());
            return make_shared<Model>();
        }

//        const shared_ptr<const Model> & ModelCache::get_model(const std::string &path, uint32_t subdiv_level) {
//            auto key = cal_key(path, subdiv_level);
//            if (is_contain(key)) {
//                return _model_map[key];
//            }
//            return load_model(path, subdiv_level);
//        }
    }
}