//
// Created by Zero on 2021/5/21.
//

#pragma once

#include "core/backend/task.h"


namespace luminous {
    inline namespace cpu {

        class CPUTask : public Task {
        private:

            UP<Integrator> _integrator;

            double _dt{0};
        public:
            CPUTask(Context *context)
                : Task(nullptr, context) {}

            void init(const Parser &parser) override;

            void update() override;

            void render_gui(double dt) override;

            void render_cli() override;

            void update_device_buffer() override;

            FrameBufferType *download_frame_buffer() override;

        };

    } // luminous::cpu
} // luminous