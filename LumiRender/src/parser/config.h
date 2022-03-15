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

        LM_ND_INLINE const std::string& full_type(const std::string &type) {
            return type;
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

        struct DebugConfig : Config {
            uint2 pixel;
        };

        struct TransformConfig : Config {
            TransformConfig() = default;

            // trs and matrix4x4 and ...
            float3 t;
            float4 r;
            float3 s;

            float4x4 mat4x4;

            float3 position;
            float yaw{};
            float pitch{};

            float3 up{};
            float3 target_pos{};

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
            bool use_normal_map{false};
            bool swap_handed{};
            uint subdiv_level{};
            // quad param
            float width{};
            float height{};

            // cube param
            float x{1},y{1},z{1};

            // mesh param
            mutable std::vector<float2> tex_coords;
            mutable std::vector<float3> positions;
            mutable std::vector<float3> normals;
            mutable std::vector<TriangleHandle> triangles;

            // sphere param
            float radius{};
            int sub_div{10};

            bool two_sided{false};
        };

        struct TextureMappingConfig : Config {
            float su{}, sv{}, du{}, dv{};
        };

        struct MaterialAttrConfig : Config {
        private:
            index_t _tex_idx{invalid_uint32};
        public:
            explicit MaterialAttrConfig(float4 v = make_float4(0.f)) : val(v) {}

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

            LM_NODISCARD bool tex_valid() const {
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

        LM_ND_INLINE bool operator==(const MaterialAttrConfig &t1, const MaterialAttrConfig &t2) {
            return t1.type() == t2.type()
                   && t1.fn == t2.fn
                   && t1.color_space == t2.color_space
                   && all(t1.val == t2.val)
                   && t1.pixel_format == t2.pixel_format;
        }

        inline bool is_contain(const std::vector<MaterialAttrConfig> &tex_configs,
                               const MaterialAttrConfig &texture_config) {
            return std::any_of(tex_configs.cbegin(), tex_configs.cend(), [&](const auto &config) {
                return config == texture_config;
            });
        }


        struct MaterialConfig : Config {

            // common
            MaterialAttrConfig color;
            bool remapping_roughness{true};
            MaterialAttrConfig roughness;

            // matte
            float sigma{};

            // assimp material
            MaterialAttrConfig specular;
            MaterialAttrConfig normal;

            // glass material
            MaterialAttrConfig eta;

            // metal material
            MaterialAttrConfig k;

            // disney material
            MaterialAttrConfig metallic;
            MaterialAttrConfig specular_tint;
            MaterialAttrConfig anisotropic;
            MaterialAttrConfig sheen;
            MaterialAttrConfig sheen_tint;
            MaterialAttrConfig clearcoat;
            MaterialAttrConfig clearcoat_roughness;
            MaterialAttrConfig spec_trans;
            MaterialAttrConfig scatter_distance;
            MaterialAttrConfig flatness;
            MaterialAttrConfig diff_trans;
            bool thin{};

            // Subsurface material, disable it temporarily
#if 0
            TextureConfig sigma_a;
            TextureConfig sigma_s;
            TextureConfig reflectance;
            TextureConfig mfp;
            float eta;
            float g;
            TextureConfig uroughness, vroughness;
            // using remapping_roughness
#endif

            static void fill_tex_idx_by_name(std::vector<MaterialAttrConfig> &tex_configs,
                                             MaterialAttrConfig &tc, bool force = false);

            void fill_tex_configs(std::vector<MaterialAttrConfig> &tex_configs);
        };

        struct OutputConfig : Config {
            std::string fn;
            EToneMap tone_map{optics::Gamma};
            int dispatch_num{};
            int frame_per_dispatch{};
            bool normal{false};
            bool albedo{false};
            bool normal_remapping{false};
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
            int tone_map{0};
            uint2 resolution{};
        };

        struct SensorConfig : Config {
            TransformConfig transform_config;
            float fov_y{};
            float velocity{};
            float focal_distance{5.f};
            float lens_radius{0.f};
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

            // for point light and spot light
            float3 intensity;
            float3 position;
            float theta_i{};
            float theta_o{};

            // for env
            MaterialAttrConfig texture_config;
            Distribution2D distribution;
            index_t distribution_idx{invalid_uint32};
            float3 scale{};
            TransformConfig o2w_config;

            void fill_tex_config(std::vector<MaterialAttrConfig> &tex_configs);
        };
    }
}

#endif