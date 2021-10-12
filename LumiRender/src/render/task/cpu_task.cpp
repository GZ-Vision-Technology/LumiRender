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

        FrameBufferType *CPUTask::get_frame_buffer() {
            _frame_buffer.synchronize_to_host();
            return _frame_buffer.data();
        }

        float4 *CPUTask::get_accumulate_buffer() {
            _accumulate_buffer.synchronize_to_host();
            return _accumulate_buffer.data();
        }

    } // luminous::cpu
} // luminous