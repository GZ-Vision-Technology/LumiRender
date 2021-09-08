//
// Created by Zero on 2021/2/21.
//


#pragma once

#include "render/include/shape.h"
#include "core/concepts.h"
#include <memory>
#include "core/context.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {

        struct SceneGraph {
        private:
            Context *_context;
            std::map <string, uint32_t> _key_to_idx;
        public:
            SamplerConfig sampler_config;
            SensorConfig sensor_config;
            std::vector <ShapeConfig> shape_configs;
            mutable std::vector <LightConfig> light_configs;
            IntegratorConfig integrator_config;
            LightSamplerConfig light_sampler_config;
            vector <SP<const Model>> model_list;
            vector <SP<const ModelInstance>> instance_list;
            vector <TextureConfig> tex_configs;
            vector <MaterialConfig> material_configs;
            OutputConfig output_config;
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