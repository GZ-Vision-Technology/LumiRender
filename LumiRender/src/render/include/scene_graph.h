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
            int spp;
        };

        struct ShapeConfig : Config {
            string type;
            struct Param {
                float4x4 o2w;
                string fn;
            };
            Param param;
        };

        struct MaterialConfig : Config {

        };

        struct SensorConfig : Config {
            string type;
            struct Param {
                float4x4 o2w;
            };
            Param param;
        };

        struct LightSamplerConfig : Config {
            string type;
        };

        struct SceneGraph {
        private:
            Context *_context;
            vector<shared_ptr<const Mesh>> mesh_list;
            vector<MeshInstance> instance_list;
        public:
            SamplerConfig sampler_config;
            SensorConfig sensor_config;
            std::vector<ShapeConfig> shape_configs;
            IntegratorConfig integrator_config;
            LightSamplerConfig light_sampler_config;

            explicit SceneGraph(Context *context) : _context(context) {}

            
            void update_mesh_list(const shared_ptr<const Model> &model);

        };

    };
}