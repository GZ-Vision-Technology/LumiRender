//
// Created by Zero on 2021/2/21.
//


#pragma once

#include "shape.h"
#include "core/concepts.h"
#include <memory>
#include "core/context.h"
#include "config.h"

namespace luminous {
    inline namespace render {

        struct SceneGraph {
        private:
            Context *_context;
            std::map <string, uint32_t> _key_to_idx;
            vector <MaterialAttrConfig> _tex_configs;
        public:
            SamplerConfig sampler_config;
            SensorConfig sensor_config;
            std::vector <ShapeConfig> shape_configs;
            mutable std::vector <LightConfig> light_configs;
            IntegratorConfig integrator_config;
            LightSamplerConfig light_sampler_config;
            vector <Model> model_list;
            vector <ModelInstance> instance_list;
            vector <MaterialConfig> material_configs;
            OutputConfig output_config;

            size_t mesh_num{};
            size_t position_num{};
            size_t tri_num{};
            size_t normal_num{};
            size_t tex_coords_num{};
            size_t instance_num{};
        private:
            bool is_contain(const string &key) {
                return _key_to_idx.find(key) != _key_to_idx.end();
            }

            void update_counter(const Model &model) {
                mesh_num += model.meshes.size();
                for (auto &mesh : model.meshes) {
                    position_num += mesh.positions.size();
                    normal_num += mesh.normals.size();
                    tex_coords_num += mesh.tex_coords.size();
                    tri_num += mesh.triangles.size();
                }
            }

        public:
            explicit SceneGraph(Context *context) : _context(context) {}

            LM_NODISCARD const vector<MaterialAttrConfig>& tex_configs() const {
                return _tex_configs;
            }

            void set_tex_configs(vector<MaterialAttrConfig>&& tcs) {
                _tex_configs = tcs;
            }

            LM_NODISCARD const MaterialAttrConfig& tex_config(int index) const {
                return _tex_configs.at(index);
            }

            LM_NODISCARD size_t try_push(const MaterialAttrConfig &tc) {
                auto key = tc.hash_key();
                for (int i = 0; i < _tex_configs.size(); ++i) {
                    const auto& elm = _tex_configs.at(i);
                    if (elm.hash_key() == tc.hash_key()) {
                        return i;
                    }
                }
                _tex_configs.push_back(tc);
                return _tex_configs.size() - 1;
            }

            Model create_shape(const ShapeConfig &config);

            Model create_quad(const ShapeConfig &config);

            Model create_quad_y(const ShapeConfig &config);

            Model create_cube(const ShapeConfig &config);

            void create_shape_instance(const ShapeConfig &config);

            void create_shapes();
        };

    };
}