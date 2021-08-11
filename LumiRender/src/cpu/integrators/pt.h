//
// Created by Zero on 2021/5/29.
//


#pragma once

#include "render/include/integrator.h"
#include "render/integrators/pt_func.h"
#include "render/samplers/sampler.h"
#include "render/sensors/sensor.h"

namespace luminous {
    inline namespace cpu {
        class CPUScene;

        class CPUPathTracer : public Integrator {
        private:
            Context *_context{};
            UP<CPUScene> _scene{nullptr};
            Sampler _sampler;
            Sensor _camera;
            int _frame_index{0};
        public:
            explicit CPUPathTracer(Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            Sensor *camera() override;

            void update() override;

            void render() override;
        };

    } // luminous::render
} // luminous