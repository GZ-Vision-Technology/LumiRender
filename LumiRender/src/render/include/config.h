//
// Created by Zero on 2021/4/14.
//

#ifndef __CUDACC__
#pragma once

#include "base_libs/math/common.h"
#include "base_libs/geometry/common.h"
#include "base_libs/optics/rgb.h"
#include <string>
#include "core/logging.h"
#include "util/image.h"
#include "base_libs/lstd/lstd.h"
#include "base_libs/sampling/distribution.h"

namespace luminous {
    inline namespace render {

        LM_ND_INLINE std::string full_type(const std::string &type) {
            return "class luminous::render::" + type;
        }

        struct Config {
        protected:
            std::string _type;
        public:
            std::string name;

            void set_type(const std::string &type) {
                _type = type;
            }

            void set_full_type(const std::string &type) {
                _type = full_type(type);
            }

            LM_NODISCARD const std::string &type() const {
                return _type;
            }
        };

        struct IntegratorConfig : Config {
            uint32_t max_depth = 10;
            float rr_threshold = 1;
        };

        struct SamplerConfig : Config {
            uint spp{};
        };

        struct TransformConfig : Config {
            TransformConfig() = default;

            // trs and matrix4x4 and ...
            float3 t;
            float4 r;
            float3 s;

            float4x4 mat4x4;

            float yaw{};
            float pitch{};
            float3 position;

            LM_NODISCARD Transform create() const {
                if (type() == "matrix4x4") {
                    return Transform(mat4x4);
                } else if (type() == "trs") {
                    auto tt = Transform::translation(t);
                    auto rr = Transform::rotation(make_float3(r), r.w);
                    auto ss = Transform::scale(s);
                    return tt * rr * ss;
                } else if (type() == "yaw_pitch") {
                    auto yaw_t = Transform::rotation_y(yaw);
                    auto pitch_t = Transform::rotation_x(pitch);
                    auto tt = Transform::translation(position);
                    return tt * pitch_t * yaw_t;
                }
#ifndef __CUDACC__
                LUMINOUS_ERROR("unknown transform type ", type());
#endif
            }
        };

        struct ShapeConfig : Config {
            ShapeConfig() = default;

            TransformConfig o2w;
            float3 emission = make_float3(0.f);

            std::string material_name;
            // model param
            mutable std::string fn;
            bool smooth{};
            bool swap_handed{};
            uint subdiv_level{};
            // quad param
            float width{};
            float height{};
        };

        struct TextureMappingConfig : Config {
            float su{}, sv{}, du{}, dv{};
        };

        struct TextureConfig : Config {
            LM_NODISCARD bool is_image() const {
                return !fn.empty();
            }

            ColorSpace color_space = LINEAR;
            index_t tex_idx{invalid_uint32};

            float3 scale{make_float3(1.f)};

            // for constant texture
            float4 val = make_float4(0.f);

            // for image texture
            index_t image_idx{invalid_uint32};
            std::string fn;
            PixelFormat pixel_format = PixelFormat::UNKNOWN;
            uint64_t handle{0};

            LM_NODISCARD bool has_image() const {
                return image_idx != invalid_uint32;
            }
        };

        LM_ND_INLINE bool operator==(const TextureConfig &t1, const TextureConfig &t2) {
            return t1.type() == t2.type()
                   && t1.fn == t2.fn
                   && t1.color_space == t2.color_space
                   && all(t1.val == t2.val)
                   && t1.pixel_format == t2.pixel_format;
        }

        inline bool is_contain(const std::vector<TextureConfig> &tex_configs,
                               const TextureConfig &texture_config) {
            return std::any_of(tex_configs.cbegin(), tex_configs.cend(), [&](const auto &config) {
                return config == texture_config;
            });
        }



        struct MaterialConfig : Config {

            // Assimp or matte

            // common
            TextureConfig diffuse_tex;

            // assimp material
            TextureConfig specular_tex;
            TextureConfig normal_tex;

            void fill_tex_configs(std::vector<TextureConfig> &tex_configs) {
                if (type() == full_type("AssimpMaterial")) {
                    int64_t index = lstd::find_index_if(tex_configs, [&](const TextureConfig &tex_config){
                        return tex_config == diffuse_tex;
                    });
                    if (index == -1) {
                        diffuse_tex.tex_idx = tex_configs.size();
                        tex_configs.push_back(diffuse_tex);
                    } else {
                        diffuse_tex.tex_idx = index;
                    }

                    index = lstd::find_index_if(tex_configs, [&](const TextureConfig &tex_config){
                        return tex_config == specular_tex;
                    });
                    if (index == -1) {
                        specular_tex.tex_idx = tex_configs.size();
                        tex_configs.push_back(specular_tex);
                    } else {
                        specular_tex.tex_idx = index;
                    }

                    index = lstd::find_index_if(tex_configs, [&](const TextureConfig &tex_config){
                        return tex_config == normal_tex;
                    });
                    if (index == -1) {
                        normal_tex.tex_idx = tex_configs.size();
                        tex_configs.push_back(normal_tex);
                    } else {
                        normal_tex.tex_idx = index;
                    }

                } else if (type() == full_type("MatteMaterial")) {
                    auto idx = lstd::find_index_if(tex_configs, [&](const TextureConfig &tex_config) {
                        return tex_config.name == diffuse_tex.name;
                    });
                    DCHECK(idx != -1);
                    diffuse_tex.tex_idx = idx;
                }
            }
        };

        struct OutputConfig : Config {
            std::string fn;
            int frame_num;
        };

        struct FilterConfig : Config {
            float2 radius;
        };

        struct FilmConfig : Config {
            int state{0};
            uint2 resolution{};
        };

        struct SensorConfig : Config {
            TransformConfig transform_config;
            float fov_y{};
            float velocity{};
            float focal_distance{};
            float lens_radius{};
            FilmConfig film_config;
        };

        struct LightSamplerConfig : Config {
        };

        class Texture;

        struct LightConfig : Config {
            LightConfig() = default;

            float3 miss_color{};
            mutable Box3f scene_box{};

            // for area light
            uint instance_idx{};
            float3 emission{};
            bool two_sided{false};
            float surface_area{1.f};

            // for point light and spot light
            float3 intensity;
            float3 position;
            float theta_i{};
            float theta_o{};

            // for env
            TextureConfig texture_config;
            Distribution2D distribution;
            index_t distribution_idx{invalid_uint32};
            float3 scale{};
            TransformConfig o2w_config;

            void fill_tex_config(std::vector<TextureConfig> &tex_configs) {
                if (type() != full_type("Envmap")) {
                    return;
                }
                int idx = lstd::find_index_if(tex_configs, [&](const TextureConfig &tex_config) {
                    return tex_config.name == texture_config.name;
                });
                texture_config = tex_configs[idx];
                texture_config.tex_idx = idx;
            }
        };
    }
}

#endif