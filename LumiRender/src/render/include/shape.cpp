//
// Created by Zero on 2021/3/7.
//

#include "shape.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/Subdivision.h>
#include <assimp/scene.h>
namespace luminous {
    inline namespace render {

        Model::Model(const std::filesystem::path &path, uint subdiv_level) {
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
            auto ai_scene = ai_importer.ReadFile(path.string().c_str(),
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

            vector<aiMesh *> ai_meshes(ai_scene->mNumMeshes);
            if (subdiv_level != 0u) {
                auto subdiv = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
                subdiv->Subdivide(ai_scene->mMeshes, ai_scene->mNumMeshes, ai_meshes.data(), subdiv_level);
            } else {
                std::copy(ai_scene->mMeshes, ai_scene->mMeshes + ai_scene->mNumMeshes, ai_meshes.begin());
            }

            meshes.reserve(ai_meshes.size());
            for (auto ai_mesh : ai_meshes) {
                vector<float3> positions;
                vector<float3> normals;
                vector<float2> tex_coords;
                positions.reserve(ai_mesh->mNumVertices);
                normals.reserve(ai_mesh->mNumVertices);
                tex_coords.reserve(ai_mesh->mNumVertices);
                vector<TriangleHandle> indices;
                indices.reserve(ai_mesh->mNumFaces);

                for (auto i = 0u; i < ai_mesh->mNumVertices; i++) {
                    auto ai_position = ai_mesh->mVertices[i];
                    auto ai_normal = ai_mesh->mNormals[i];
                    auto position = make_float3(ai_position.x, ai_position.y, ai_position.z);
                    auto normal = make_float3(ai_normal.x, ai_normal.y, ai_normal.z);
                    if (ai_mesh->mTextureCoords[0] != nullptr) {
                        auto ai_tex_coord = ai_mesh->mTextureCoords[0][i];
                        auto uv = make_float2(ai_tex_coord.x, ai_tex_coord.y);
                        tex_coords.push_back(uv);
                    } else {
                        tex_coords.emplace_back(0,0);
                    }
                    positions.push_back(position);
                    normals.push_back(normal);
                }

                for (auto f = 0u; f < ai_mesh->mNumFaces; f++) {
                    auto ai_face = ai_mesh->mFaces[f];
                    if (ai_face.mNumIndices == 3) {
                        indices.push_back(TriangleHandle{
                                ai_face.mIndices[0],
                                ai_face.mIndices[1],
                                ai_face.mIndices[2]});
                    } else if (ai_face.mNumIndices == 4) {
                        indices.push_back(TriangleHandle{
                                ai_face.mIndices[0],
                                ai_face.mIndices[1],
                                ai_face.mIndices[2]});
                        indices.push_back(TriangleHandle{
                                ai_face.mIndices[0],
                                ai_face.mIndices[2],
                                ai_face.mIndices[3]});
                    } else {
                        LUMINOUS_EXCEPTION("Only triangles and quads supported: ", ai_mesh->mName.data);
                    }
                }
                auto mesh = std::make_shared<const Mesh>(move(positions),
                                                    move(normals),
                                                    move(tex_coords),
                                                    move(indices));
                meshes.push_back(mesh);
            }

        } // luminous::Model
    } // luminous::render
} // luminous