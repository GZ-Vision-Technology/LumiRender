//
// Created by Zero on 2021/2/21.
//


#pragma once

#include "shape.h"
#include "material.h"
#include "render/lights/light.h"
#include "render/light_samplers/light_sampler.h"
#include "core/concepts.h"
#include <memory>
#include "core/context.h"

namespace luminous {
    using namespace std;
    inline namespace render {
        struct Config {
        };

        struct IntegratorConfig : Config {
            string type;
        };

        struct SamplerConfig : Config {
            string type;
            uint spp{};
        };

        struct MaterialConfig : Config {

        };

        struct TransformConfig : Config {
            // trs and matrix4x4 and ...
            string type;
            // trs
            float3 t;
            float4 r;
            float3 s;
            // matrix
            float4x4 mat4x4;
            // yaw_pitch
            float yaw;
            float pitch;
            float3 position;

            Transform create() const {
                if (type == "matrix4x4") {
                    return Transform(mat4x4);
                } else if (type == "trs") {
                    auto tt = Transform::translation(t);
                    auto rr = Transform::rotation(make_float3(r), r.w);
                    auto ss = Transform::scale(s);
                    return tt * rr * ss;
                } else if (type == "yaw_pitch") {
                    auto yaw_t = Transform::rotation_y(yaw);
                    auto pitch_t = Transform::rotation_x(pitch);
                    auto tt = Transform::translation(position);
                    return tt * pitch_t * yaw_t;
                }
                LUMINOUS_ERROR("unknown transform type ", type);
            }
        };

        struct ShapeConfig : Config {
            string type;
            string fn;
            string name;
            TransformConfig o2w;
            uint subdiv_level;

            float width;
            float height;
        };

        struct FilterConfig {
            string type;
            float2 radius;
        };

        struct FilmConfig : Config {
            string type;
            uint2 resolution;
            string file_name;
        };

        struct SensorConfig : Config {
            string type;
            TransformConfig transform_config;
            float fov_y;
            float velocity;
            FilmConfig film_config;
        };

        struct LightSamplerConfig : Config {
            string type;
        };

        struct SceneGraph {
        private:
            Context *_context;
            std::map<string, uint32_t> _key_to_idx;
        public:
            SamplerConfig sampler_config;
            SensorConfig sensor_config;
            std::vector<ShapeConfig> shape_configs;
            IntegratorConfig integrator_config;
            LightSamplerConfig light_sampler_config;
            vector<SP<const Model>> model_list;
            vector<SP<const ModelInstance>> instance_list;
        private:
            bool is_contain(const string &key) {
                return _key_to_idx.find(key) != _key_to_idx.end();
            }
        public:
            explicit SceneGraph(Context *context) : _context(context) {}

            SP<Model> create_shape(const ShapeConfig &config);

            void create_shape_instance(const ShapeConfig &config);

            void create_shapes();
        };

    };
}