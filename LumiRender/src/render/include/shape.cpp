//
// Created by Zero on 2021/3/7.
//

#include "shape.h"
#include "util/assimp_parser.h"
#include "render/textures/texture.h"

namespace luminous {
    inline namespace render {

        std::pair<string, float4> load_texture(const aiMaterial *mat, aiTextureType type) {
            string fn = "";
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
                mc.diffuse_tex.color_space = LINEAR;
            }
            {
                // process specular
                auto[specular_fn, specular] = load_texture(ai_material, aiTextureType_SPECULAR);
                mc.specular_tex.fn = model->full_path(specular_fn);
                mc.specular_tex.val = specular;
                mc.specular_tex.name = "specular";
                auto tex_type = mc.specular_tex.fn.empty() ? type_name<ConstantTexture>() : type_name<ImageTexture>();
                mc.specular_tex.set_type(tex_type);
                mc.specular_tex.color_space = LINEAR;
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
            if (model->has_custom_material()) {
                return;
            }
            vector<aiMaterial *> ai_materials(ai_scene->mNumMaterials);
            model->materials.reserve(ai_materials.size());
            std::copy(ai_scene->mMaterials, ai_scene->mMaterials + ai_scene->mNumMaterials, ai_materials.begin());
            for (const auto &ai_material : ai_materials) {
                MaterialConfig mc = process_material(ai_material, model);
                mc.set_full_type("AssimpMaterial");
                model->materials.push_back(mc);
            }
        }

        Model::Model(const ShapeConfig &sc) {
            Assimp::Importer ai_importer;
            luminous_fs::path path = sc.fn;
            directory = path.parent_path();

            auto ai_scene = AssimpParser::load_scene(sc.fn, ai_importer, sc.swap_handed, sc.smooth);

            LUMINOUS_EXCEPTION_IF(
                    ai_scene == nullptr || (ai_scene->mFlags & static_cast<uint>(AI_SCENE_FLAGS_INCOMPLETE)) ||
                    ai_scene->mRootNode == nullptr,
                    "Failed to load triangle mesh: ", ai_importer.GetErrorString());

            meshes = AssimpParser::parse_meshes(ai_scene, sc.subdiv_level);

            custom_material_name = sc.material_name;
            process_materials(ai_scene, this);
        }
    } // luminous::render
} // luminous