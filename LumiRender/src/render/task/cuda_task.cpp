//
// Created by Zero on 2021/2/18.
//

#include "cuda_task.h"
#include "util/parser.h"
#include "render/integrators/megakernel_pt.h"
#include "render/integrators/wavefront/integrator.h"

using std::cout;
using std::endl;
namespace luminous {
    inline namespace gpu {

        void CUDATask::init(const Parser &parser) {
            auto scene_graph = build_scene_graph(parser);
            const std::string type = scene_graph->integrator_config.type();
            if (type == "PT") {
                _integrator = std::make_unique<MegakernelPT>(_device.get(), _context);
            } else if(type == "WavefrontPT") {
                _integrator = std::make_unique<WavefrontPT>(_device.get(), _context);
            }
            _integrator->init(scene_graph);
            update_device_buffer();
        }
    }
}