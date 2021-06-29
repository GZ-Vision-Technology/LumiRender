//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "render/include/scene_graph.h"
#include "render/sensors/sensor.h"

namespace luminous {
    inline namespace render {
        class Integrator : public Noncopyable {
        protected:
            uint _max_depth;
            float _rr_threshold;
        public:
            virtual ~Integrator() {}

            virtual void init(const std::shared_ptr<SceneGraph> &scene_graph) = 0;

            virtual void test() {}

            virtual void init_with_config(const IntegratorConfig &ic) {
                _max_depth = ic.max_depth;
                _rr_threshold = ic.rr_threshold;
            }

            virtual Sensor *camera() = 0;

            virtual void update() = 0;

            virtual void render() = 0;
        };
    }
}