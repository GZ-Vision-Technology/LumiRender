//
// Created by Zero on 2021/5/21.
//

#include "cpu_task.h"
#include "util/parser.h"
#include "render/integrators/cpu_pt.h"
#include "render/integrators/wavefront/integrator.h"

namespace luminous {
    inline namespace cpu {
        void CPUTask::init(const Parser &parser) {
            auto scene_graph = build_scene_graph(parser);
            const std::string type = scene_graph->integrator_config.type();
            if (type == "PT") {
                _integrator = std::make_unique<CPUPathTracer>(_device.get(), _context);
            } else if(type == "WavefrontPT") {
                _integrator = std::make_unique<WavefrontPT>(_device.get(), _context);
            }
            _integrator->init(scene_graph);
            update_device_buffer();
        }

        void CPUTask::render_cli() {

        }

        void CPUTask::update_device_buffer() {
            auto res = camera()->film()->resolution();
            auto num = res.x * res.y;
            _accumulate_buffer.resize(num, make_float4(0.f));
            _accumulate_buffer.allocate_device(num);
            camera()->film()->set_accumulate_buffer_view(_accumulate_buffer.device_buffer_view());

            _frame_buffer.reset(num);
            _frame_buffer.synchronize_to_device();
            camera()->film()->set_frame_buffer_view(_frame_buffer.device_buffer_view());
        }

        FrameBufferType *CPUTask::get_frame_buffer() {
            return _frame_buffer.data();
        }

        float4 *CPUTask::get_accumulate_buffer() {
            return _accumulate_buffer.data();
        }

    } // luminous::cpu
} // luminous