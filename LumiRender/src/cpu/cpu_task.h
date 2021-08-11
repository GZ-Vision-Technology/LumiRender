//
// Created by Zero on 2021/5/21.
//

#pragma once

#include "core/backend/task.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace cpu {

        class CPUTask : public Task {
        public:
            explicit CPUTask(Context *context)
                : Task(nullptr, context) {}

            void init(const Parser &parser) override;

            void render_gui(double dt) override;

            void render_cli() override;

            void update_device_buffer() override;

            FrameBufferType *download_frame_buffer() override;

        };

    } // luminous::cpu
} // luminous