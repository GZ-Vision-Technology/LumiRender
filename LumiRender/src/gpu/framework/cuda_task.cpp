//
// Created by Zero on 2021/2/18.
//

#include "cuda_task.h"
#include "gpu/integrators/megakernel_pt.h"
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

        void CUDATask::update_device_buffer() {
            auto res = camera()->film()->resolution();
            auto num = res.x * res.y;
            _accumulate_buffer.allocate_device(num);
            camera()->film()->set_accumulate_buffer_view(_accumulate_buffer.device_buffer_view());

            _frame_buffer.reset(num);
            _frame_buffer.synchronize_to_device();
            camera()->film()->set_frame_buffer_view(_frame_buffer.device_buffer_view());
        }

        FrameBufferType *CUDATask::get_frame_buffer() {
            _frame_buffer.synchronize_to_host();
            return _frame_buffer.data();
        }

        float4 *CUDATask::get_accumulate_buffer() {
            _accumulate_buffer.synchronize_to_host();
            return _accumulate_buffer.data();
        }
    }
}