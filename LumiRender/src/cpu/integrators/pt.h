//
// Created by Zero on 2021/5/29.
//


#pragma once

#include "render/integrators/integrator.h"
#include "render/samplers/sampler.h"
#include "render/sensors/sensor.h"

namespace luminous {
    inline namespace cpu {
        class CPUScene;

        class CPUPathTracer : public Integrator {
        private:
            int _frame_index{0};
        public:
            CPUPathTracer(const SP<Device> &device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            template<typename TScene>
            NDSC decltype(auto) scene() {
                return reinterpret_cast<TScene*>(_scene.get());
            }

            NDSC uint frame_index() const override { return _frame_index; }

            void update() override;

            void render() override;
        };

    } // luminous::render
} // luminous