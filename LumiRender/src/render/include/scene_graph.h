//
// Created by Zero on 2021/2/21.
//


#pragma once

#include "sensor.h"
#include "shape.h"
#include "material.h"
#include "light.h"
#include "light_sampler.h"
#include "core/concepts.h"
#include <memory>
#include "core/context.h"
#include "model.h"

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
            int spp{};
        };

        struct MaterialConfig : Config {

        };

        struct TransformConfig : Config {
            // trs and matrix4x4
            string type;
            float4 vec4;
            float3 t;
            float4 r;
            float3 s;
            float4x4 mat4x4;
        };

        struct ShapeConfig : Config {
            string type;
            string fn;
            TransformConfig o2w;
        };

        struct FilterConfig {
            string type;
            uint2 radius;
        };

        struct FilmConfig : Config {
            string type;
            uint2 resolution;
        };

        struct SensorConfig : Config {
            string type;
            TransformConfig transform_config;
        };

        struct LightSamplerConfig : Config {
            string type;
        };

        struct SceneGraph {
        private:
            Context *_context;
            vector<shared_ptr<const Mesh>> _mesh_list;
            vector<MeshInstance> _instance_list;
        public:
            SamplerConfig sampler_config;
            SensorConfig sensor_config;
            std::vector<ShapeConfig> shape_configs;
            IntegratorConfig integrator_config;
            LightSamplerConfig light_sampler_config;

            explicit SceneGraph(Context *context) : _context(context) {}
        };

    };
}