//
// Created by Zero on 2021/3/7.
//

#include "shape.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/Subdivision.h>
#include <assimp/scene.h>
#include "render/textures/texture.h"

namespace luminous {
    inline namespace render {

        std::pair<string, float4> load_texture(const aiMaterial *mat, aiTextureType type) {
            string fn = "";
            for(size_t i = 0; i < mat->GetTextureCount(type); ++i) {
                aiString str;
                mat->GetTexture(type, i, &str);
                fn = str.C_Str();
                break;
            }
            aiColor3D ai_color;
            switch (type) {
                case aiTextureType_DIFFUSE:
                    mat->Get(AI_MATKEY_COLOR_DIFFUSE, ai_color);
                    break;
                case aiTextureType_SPECULAR:
                    mat->Get(AI_MATKEY_COLOR_SPECULAR, ai_color);
                    break;
                default:
                    break;
            }
            float4 color = make_float4(ai_color.r, ai_color.g, ai_color.b, 0);
            return make_pair(fn, color);
        }

        MaterialConfig process_material(const aiMaterial *ai_material, Model *model) {
            MaterialConfig mc;
            {
                // process diffuse
                auto[diffuse_fn, diffuse] = load_texture(ai_material, aiTextureType_DIFFUSE);
                mc.diffuse_tex.fn = model->full_path(diffuse_fn);
                auto tex_type = mc.diffuse_tex.fn.empty() ? type_name<ConstantTexture>() : type_name<ImageTexture>();
                mc.diffuse_tex.val = diffuse;
                mc.diffuse_tex.name = "diffuse";
                mc.diffuse_tex.set_type(tex_type);
                mc.diffuse_tex.color_space = SRGB;
            }
            {
                // process specular
                auto[specular_fn, specular] = load_texture(ai_material, aiTextureType_SPECULAR);
                mc.specular_tex.fn = model->full_path(specular_fn);
                mc.specular_tex.val = specular;
                mc.specular_tex.name = "specular";
                auto tex_type = mc.specular_tex.fn.empty() ? type_name<ConstantTexture>() : type_name<ImageTexture>();
                mc.specular_tex.set_type(tex_type);
                mc.specular_tex.color_space = SRGB;
            }
            {
                // process normal map
                auto[normal_fn, _] = load_texture(ai_material, aiTextureType_HEIGHT);
                mc.normal_tex.set_type(type_name<ImageTexture>());
                mc.normal_tex.name = "normal";
                mc.normal_tex.fn = model->full_path(normal_fn);
                mc.normal_tex.color_space = LINEAR;
            }
            return mc;
        }

        void process_materials(const aiScene *ai_scene, Model *model) {
            vector<aiMaterial*> ai_materials(ai_scene->mNumMaterials);
            model->materials.reserve(ai_materials.size());
            std::copy(ai_scene->mMaterials, ai_scene->mMaterials + ai_scene->mNumMaterials, ai_materials.begin());
            for(const auto &ai_material : ai_materials) {
                MaterialConfig mc = process_material(ai_material, model);
                mc.set_full_type("AssimpMaterial");
                model->materials.push_back(mc);
            }
        }

        Model::Model(const std::filesystem::path &path, uint subdiv_level, bool smooth) {
            Assimp::Importer ai_importer;
            directory = path.parent_path();
            ai_importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                           aiComponent_COLORS |
                                           aiComponent_BONEWEIGHTS |
                                           aiComponent_ANIMATIONS |
                                           aiComponent_LIGHTS |
                                           aiComponent_CAMERAS |
                                           aiComponent_TEXTURES |
                                           aiComponent_MATERIALS);
            LUMINOUS_INFO("Loading triangle mesh: ", path);
            aiPostProcessSteps normal_flag = smooth ? aiProcess_GenSmoothNormals : aiProcess_GenNormals;
            auto ai_scene = ai_importer.ReadFile(path.string().c_str(),
                                                 aiProcess_JoinIdenticalVertices |
                                                 normal_flag |
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

            process_materials(ai_scene, this);

            meshes.reserve(ai_meshes.size());
            for (auto ai_mesh : ai_meshes) {
                Box3f aabb;
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
                    aabb.extend(position);
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
                                                    move(indices),
                                                    aabb,
                                                    ai_mesh->mMaterialIndex);
                meshes.push_back(mesh);
            }
        }
    } // luminous::render
} // luminous