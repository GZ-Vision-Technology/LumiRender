//
// Created by Zero on 2021/4/14.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/geometry/common.h"
#include "graphics/optics/rgb.h"
#include <string>
#include "core/logging.h"
#include "util/image_base.h"

namespace luminous {
    inline namespace render {

        NDSC_INLINE std::string full_type(const std::string &type) {
            return "class luminous::render::" + type;
        }

        struct Config {
        protected:
            std::string _type;
        public:
            void set_type(const std::string &type) {
                _type = type;
            }

            void set_full_type(const std::string &type) {
                _type = full_type(type);
            }

            const std::string &type() const {
                return _type;
            }
        };

        struct IntegratorConfig : Config {
        };

        struct SamplerConfig : Config {
            uint spp{};
        };

        struct TransformConfig : Config {
            TransformConfig() {}

            // trs and matrix4x4 and ...
            union {
                struct {
                    // trs
                    float3 t;
                    float4 r;
                    float3 s;
                };
                struct {
                    float4x4 mat4x4;
                };
                struct {
                    float yaw;
                    float pitch;
                    float3 position;
                };
            };

            Transform create() const {
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
                LUMINOUS_ERROR("unknown transform type ", type());
            }
        };

        struct ShapeConfig : Config {
            ShapeConfig() {
            }

            std::string name;
            TransformConfig o2w;
            float3 emission = make_float3(0.f);
//            union {
//                // model param
//                struct {
            std::string fn;
            uint subdiv_level;
//                };
//                // quad param
//                struct {
            float width;
            float height;
//                };
//            };
        };

        struct TextureMappingConfig : Config {
            float su, sv, du, dv;
        };

        struct TextureConfig : Config {
            bool is_image() const {
                return fn != "";
            }

            std::string name;

            ColorSpace color_space;
            // for constant texture
            float4 val;
            // for image texture
            std::string fn = "";
            PixelFormat pixel_format;
            void *handle{nullptr};
        };

        struct MaterialConfig : Config {

            // common
            float4 diffuse;
            string diffuse_fn;
            index_t diffuse_idx{index_t(-1)};

            // assimp material
            float4 specular;
            string specular_fn;
            index_t specular_idx{index_t(-1)};
            string normal_fn;
            index_t normal_idx{index_t(-1)};
        };

        struct FilterConfig : Config {
            float2 radius;
        };

        struct FilmConfig : Config {
            uint2 resolution;
            std::string file_name;
        };

        struct SensorConfig : Config {
            TransformConfig transform_config;
            float fov_y;
            float velocity;
            float focal_distance;
            float lens_radius;
            FilmConfig film_config;
        };

        struct LightSamplerConfig : Config {
        };

        struct LightConfig : Config {
            LightConfig() {}

            union {
                struct {
                    uint instance_idx;
                    float3 emission;
                };
                struct {
                    float3 intensity;
                    float3 position;
                };
            };
        };
    }
}