//
// Created by Zero on 30/12/2021.
//

#include "config.h"
#include "core/logging.h"
#include "util/image.h"

namespace luminous {
    inline namespace render {

        Transform TransformConfig::create() const {
            if (type() == "matrix4x4") {
                return Transform(mat4x4);
            } else if (type() == "trs") {
                auto tt = Transform::translation(t);
                auto rr = Transform::rotation(make_float3(r), r.w);
                auto ss = Transform::scale(s);
                return tt * rr * ss;
            } else if (type() == "yaw_pitch") {
                auto yaw_t = Transform::rotation_y(yaw);
                auto pitch_t = Transform::rotation_x(-pitch);
                auto tt = Transform::translation(position);
                return tt * pitch_t * yaw_t;
            }
            LUMINOUS_ERROR("unknown transform type ", type());
        }

        void MaterialConfig::fill_tex_idx_by_name(vector<MaterialAttrConfig> &tex_configs, MaterialAttrConfig &tc,
                                                  bool force) {
            if (tc.name.empty()) {
                return;
            }
            auto idx = lstd::find_index_if(tex_configs, [&](const MaterialAttrConfig &tex_config) {
                return tex_config.name == tc.name;
            });
            tc.fill_tex_idx(idx, force);
        }

        void MaterialConfig::fill_tex_configs(vector<MaterialAttrConfig> &tex_configs) {
            // common data
            if (!normal.name.empty()) {
                fill_tex_idx_by_name(tex_configs, normal, false);
            }

            if (type() == full_type("AssimpMaterial")) {
                if (!color.fn.empty()) {
                    color.set_tex_idx(tex_configs.size());
                    tex_configs.push_back(color);
                }

                if (!specular.fn.empty()) {
                    specular.set_tex_idx(tex_configs.size());
                    tex_configs.push_back(specular);
                }

                if (!normal.fn.empty()) {
                    normal.set_tex_idx(tex_configs.size());
                    tex_configs.push_back(normal);
                }

            } else if (type() == full_type("MatteMaterial")) {

                fill_tex_idx_by_name(tex_configs, color);

            } else if (type() == full_type("MetalMaterial")) {

                fill_tex_idx_by_name(tex_configs, roughness);

                fill_tex_idx_by_name(tex_configs, k);

                fill_tex_idx_by_name(tex_configs, eta);

            } else if (type() == full_type("GlassMaterial")) {
                fill_tex_idx_by_name(tex_configs, color);

                fill_tex_idx_by_name(tex_configs, roughness);

                fill_tex_idx_by_name(tex_configs, eta);

            } else if (type() == full_type("FakeMetalMaterial")) {

                fill_tex_idx_by_name(tex_configs, color);
                fill_tex_idx_by_name(tex_configs, roughness);

            } else if (type() == full_type("MirrorMaterial")) {
                fill_tex_idx_by_name(tex_configs, color);
            } else if (type() == full_type("DisneyMaterial")) {

                fill_tex_idx_by_name(tex_configs, color);
                fill_tex_idx_by_name(tex_configs, roughness);
                fill_tex_idx_by_name(tex_configs, eta);
                fill_tex_idx_by_name(tex_configs, metallic);
                fill_tex_idx_by_name(tex_configs, specular_tint);
                fill_tex_idx_by_name(tex_configs, anisotropic);
                fill_tex_idx_by_name(tex_configs, sheen);
                fill_tex_idx_by_name(tex_configs, sheen_tint);
                fill_tex_idx_by_name(tex_configs, clearcoat);
                fill_tex_idx_by_name(tex_configs, clearcoat_roughness);
                fill_tex_idx_by_name(tex_configs, spec_trans);
                fill_tex_idx_by_name(tex_configs, scatter_distance);
                fill_tex_idx_by_name(tex_configs, flatness);
                fill_tex_idx_by_name(tex_configs, diff_trans);

            }
        #if 0
        // Disable SubsurfaceMaterial parsing temporarily
        else if(type() == full_type("SubsurfaceMaterial")) {
            fill_tex_idx_by_name(tex_configs, sigma_s);
            fill_tex_idx_by_name(tex_configs, sigma_a);
            fill_tex_idx_by_name(tex_configs, reflectance);
            fill_tex_idx_by_name(tex_configs, mfp);
            fill_tex_idx_by_name(tex_configs, uroughness);
            fill_tex_idx_by_name(tex_configs, vroughness);
        }
        #endif
        }

        void LightConfig::fill_tex_config(vector<MaterialAttrConfig> &tex_configs) {
            if (type() != full_type("Envmap")) {
                return;
            }
            int idx = lstd::find_index_if(tex_configs, [&](const MaterialAttrConfig &tex_config) {
                return tex_config.name == texture_config.name;
            });
            texture_config = tex_configs[idx];
            texture_config.fill_tex_idx(idx);
        }
    }
}