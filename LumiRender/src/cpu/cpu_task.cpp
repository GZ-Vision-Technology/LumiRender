//
// Created by Zero on 2021/5/21.
//

#include "cpu_task.h"

namespace luminous {
    inline namespace cpu {
        void CPUTask::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
//            _integrator = make_unique<MegakernelPT>(_device, _context);
//            _integrator->init(scene_graph);
//            update_device_buffer();
        }

        void CPUTask::render_gui(double dt) {

        }

        void CPUTask::render_cli() {

        }

        void CPUTask::update_device_buffer() {
            auto res = camera()->film()->resolution();
            auto num = res.x * res.y;
            _accumulate_buffer.resize(num, make_float4(0.f));
            camera()->film()->set_accumulate_buffer_view(_accumulate_buffer.host_buffer_view());

            _frame_buffer.resize(num, 0);
            camera()->film()->set_frame_buffer_view(_frame_buffer.host_buffer_view());
        }

        FrameBufferType *CPUTask::download_frame_buffer() {
            return nullptr;
        }

    } // luminous::cpu
} // luminous