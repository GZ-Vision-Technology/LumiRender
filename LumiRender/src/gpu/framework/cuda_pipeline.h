//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/pipeline.h"
#include "cuda_device.h"

namespace luminous {
    inline namespace gpu {
        class CUDAPipeline : public Pipeline {
        private:
            unique_ptr<Scene> _scene{nullptr};
        public:
            CUDAPipeline(unique_ptr<CUDADevice> cuda_device, Context *context)
                : Pipeline(make_unique<Device>(move(cuda_device)),context) {}

            void init(const Parser &parser) override {
                auto scene_graph = parser.parse();
                scene_graph->create_scene();
                _scene = make_unique<Scene>(move(scene_graph));
            }

            void render_cli() override {

            }

            void render_gui() override {

            }
        };
    }
}