//
// Created by Zero on 2021/5/29.
//


#pragma once

#include "render/integrators/integrator.h"
#include "render/integrators/pt_func.h"
#include "render/samplers/sampler.h"
#include "render/sensors/sensor.h"

namespace luminous {
    inline namespace cpu {
        class CPUScene;

        class CPUPathTracer : public Integrator {
        private:
            UP<CPUScene> _scene{nullptr};
            Sampler _sampler;
            Sensor _camera;
            int _frame_index{0};
        public:
            CPUPathTracer(const SP<Device> &device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            NDSC uint frame_index() const override {
                return _frame_index;
            }

            Sensor *camera() override;

            void update() override;

            void render() override;
        };

    } // luminous::render
} // luminous