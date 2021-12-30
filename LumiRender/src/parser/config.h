//
// Created by Zero on 2021/4/14.
//

#ifndef __CUDACC__
#pragma once

#include "base_libs/math/common.h"
#include "base_libs/geometry/common.h"
#include "base_libs/optics/rgb.h"
#include "util/image_base.h"
#include <string>
#include "core/hash.h"
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
            uint32_t min_depth = 5;
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

            LM_NODISCARD Transform create() const;
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
            // mesh param
            mutable std::vector<float2> tex_coords;
            mutable std::vector<float3> positions;
            mutable std::vector<float3> normals;
            mutable std::vector<TriangleHandle> triangles;
        };

        struct TextureMappingConfig : Config {
            float su{}, sv{}, du{}, dv{};
        };

        struct TextureConfig : Config {
        private:
            index_t _tex_idx{invalid_uint32};
        public:
            LM_NODISCARD bool is_image() const {
                return !fn.empty();
            }

            LM_NODISCARD index_t tex_idx() const {
                return _tex_idx;
            }

            void fill_tex_idx(index_t index, bool force = false) {
                if (force || !is_valid_index(_tex_idx)) {
                    _tex_idx = index;
                }
            }

            LM_NODISCARD bool valid() const {
                return is_valid_index(_tex_idx);
            }

            void set_tex_idx(index_t index) {
                _tex_idx = index;
            }

            ColorSpace color_space = LINEAR;

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

            LM_NODISCARD SHA1 hash_key() const {
                std::string str = fn +
                                  "type:" + type() +
                                  ",val:" + val.to_string() +
                                  ",scale:" + scale.to_string() +
                                  ",color space:" + std::to_string(int(color_space));
                return SHA1(str);
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

            // common
            TextureConfig color_tex;
            bool remapping_roughness{true};
            TextureConfig roughness_tex;

            // matte
            float sigma{};

            // assimp material
            TextureConfig specular_tex;
            TextureConfig normal_tex;

            // glass material
            TextureConfig eta_tex;

            // metal material
            TextureConfig k_tex;

            // disney material
            TextureConfig metallic_tex;
            TextureConfig specular_tint_tex;
            TextureConfig anisotropic_tex;
            TextureConfig sheen_tex;
            TextureConfig sheen_tint_tex;
            TextureConfig clearcoat_tex;
            TextureConfig clearcoat_gloss_tex;
            TextureConfig spec_trans_tex;
            TextureConfig scatter_distance_tex;
            TextureConfig flatness_tex;
            TextureConfig diff_trans_tex;
            bool thin{};

            static void fill_tex_idx_by_name(std::vector<TextureConfig> &tex_configs,
                                             TextureConfig &tc, bool force = false);

            void fill_tex_configs(std::vector<TextureConfig> &tex_configs);
        };

        struct OutputConfig : Config {
            std::string fn;
            int dispatch_num{};
            int frame_per_dispatch{};
        };

        struct FilterConfig : Config {
            float2 radius;

            // for gaussian filter
            float sigma{};

            // for sinc filter
            float tau{};

            // for mitchell filter
            float b{}, c{};
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
            FilterConfig filter_config;
        };

        struct LightSamplerConfig : Config {
        };

        class Texture;

        struct LightConfig : Config {
            LightConfig() = default;

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

            void fill_tex_config(std::vector<TextureConfig> &tex_configs);
        };
    }
}

#endif