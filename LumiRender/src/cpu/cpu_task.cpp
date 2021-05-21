//
// Created by Zero on 2021/5/21.
//

#include "cpu_task.h"
#include ""

namespace luminous {
    inline namespace cpu {
        void CPUTask::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
//            _integrator = make_unique<MegakernelPT>(_device, _context);
//            _integrator->init(scene_graph);
//            update_device_buffer();
        }
    } // luminous::cpu
} // luminous