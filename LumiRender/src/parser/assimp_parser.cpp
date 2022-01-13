//
// Created by Zero on 15/11/2021.
//


#include "assimp_parser.h"
#include "shape.h"
#include "render/textures/texture.h"

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
                                      aiProcess_FindDegenerates |
                                      aiProcess_GenUVCoords |
                                      aiProcess_TransformUVCoords |
                                      aiProcess_OptimizeMeshes |
                                      aiProcess_Triangulate |
                                      aiProcess_FlipUVs;
            post_process_steps = swap_handed ?
                                 post_process_steps | aiProcess_ConvertToLeftHanded :
                                 post_process_steps;
            auto ai_scene = ai_importer.ReadFile(fn.string().c_str(),
                                                 post_process_steps);

            return ai_scene;
        }

        std::vector<Mesh> AssimpParser::parse_meshes(const aiScene *ai_scene, uint32_t subdiv_level) {
            std::vector<Mesh> meshes;
            vector<aiMesh *> ai_meshes(ai_scene->mNumMeshes);
            if (subdiv_level != 0u) {
                auto subdiv = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
                subdiv->Subdivide(ai_scene->mMeshes, ai_scene->mNumMeshes, ai_meshes.data(), subdiv_level);
            } else {
                std::copy(ai_scene->mMeshes, ai_scene->mMeshes + ai_scene->mNumMeshes, ai_meshes.begin());
            }
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
                        tex_coords.emplace_back(0.f, 0.f);
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
                        LUMINOUS_EXCEPTION("Only triangles and quads supported: ", ai_mesh->mName.data, " num is ",
                                           ai_face.mNumIndices);
                    }
                }
                auto mesh = Mesh(move(positions),
                                 move(normals),
                                 move(tex_coords),
                                 move(indices),
                                 aabb,
                                 ai_mesh->mMaterialIndex);
                meshes.push_back(mesh);
            }
            return meshes;
        }

        std::vector<MaterialConfig> AssimpParser::parse_materials(const aiScene *ai_scene,
                                                                  const luminous_fs::path &directory,
                                                                  bool use_normal_map) {
            std::vector<MaterialConfig> ret;
            vector<aiMaterial *> ai_materials(ai_scene->mNumMaterials);
            ret.reserve(ai_materials.size());
            std::copy(ai_scene->mMaterials, ai_scene->mMaterials + ai_scene->mNumMaterials, ai_materials.begin());
            for (const auto &ai_material : ai_materials) {
                MaterialConfig mc = parse_material(ai_material, directory, use_normal_map);
                mc.set_full_type("AssimpMaterial");
                ret.push_back(mc);
            }
            return ret;
        }

        MaterialConfig AssimpParser::parse_material(const aiMaterial *ai_material,
                                                    const luminous_fs::path &directory,
                                                    bool use_normal_map) {
            auto full_path = [&](const luminous_fs::path &fn) -> std::string {
                return fn.empty() ? fn.string() : (directory / fn).string();
            };
            MaterialConfig mc;
            {
                // process diffuse
                auto[diffuse_fn, diffuse] = load_texture(ai_material, aiTextureType_DIFFUSE);
                mc.color_tex.fn = full_path(diffuse_fn);
                auto tex_type = mc.color_tex.fn.empty() ? type_name<ConstantTexture>() : type_name<ImageTexture>();
                mc.color_tex.val = diffuse;
                mc.color_tex.name = "color";
                mc.color_tex.set_type(tex_type);
                mc.color_tex.color_space = LINEAR;
            }
            {
                // process specular
                auto[specular_fn, specular] = load_texture(ai_material, aiTextureType_SPECULAR);
                mc.specular_tex.fn = full_path(specular_fn);
                mc.specular_tex.val = specular;
                mc.specular_tex.name = "specular";
                auto tex_type = mc.specular_tex.fn.empty() ? type_name<ConstantTexture>() : type_name<ImageTexture>();
                mc.specular_tex.set_type(tex_type);
                mc.specular_tex.color_space = LINEAR;
            }
            if (use_normal_map) {
                auto[normal_fn, normal] = load_texture(ai_material, aiTextureType_HEIGHT);
                if (normal_fn.empty()) {
                    return mc;
                }
                mc.normal_tex.set_type(type_name<ImageTexture>());
                mc.normal_tex.name = "normal";
                mc.normal_tex.val = normal;
                mc.normal_tex.fn = full_path(normal_fn);
                auto tex_type = mc.normal_tex.fn.empty() ? type_name<ConstantTexture>() : type_name<ImageTexture>();
                mc.normal_tex.set_type(tex_type);
                mc.normal_tex.color_space = LINEAR;
            }
            return mc;

        }

        std::pair<string, float4> AssimpParser::load_texture(const aiMaterial *mat, aiTextureType type) {
            string fn;
            for (size_t i = 0; i < mat->GetTextureCount(type); ++i) {
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
            return std::make_pair(fn, color);
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
            auto scene_graph = std::make_shared<SceneGraph>(_context);

            scene_graph->light_configs = parse_lights();
            scene_graph->sensor_config = parse_camera();

            return scene_graph;
        }

        std::vector<LightConfig> AssimpParser::parse_lights() const {
            std::vector<LightConfig> ret;
            ret.reserve(_ai_scene->mNumLights);

            for (int i = 0; i < _ai_scene->mNumLights; ++i) {
                auto lc = parse_light(_ai_scene->mLights[i]);
                ret.push_back(lc);
            }

            return ret;
        }

        LightConfig AssimpParser::parse_light(const aiLight *ai_light) const {
            LightConfig ret;

            return ret;
        }

        SensorConfig AssimpParser::parse_camera() const {
            SensorConfig ret;
            return ret;
        }
    }
}