//
// Created by Zero on 2021/2/18.
//

#include "cuda_task.h"
#include "util/clock.h"
#include "gpu/megakernel_pt.h"

namespace luminous {
    inline namespace gpu {

        void CUDATask::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
            _integrator = make_unique<MegakernelPT>(_device, _context);
            _integrator->init(scene_graph);
            update_device_buffer();
        }

        void CUDATask::update_device_buffer() {
            auto res = camera()->film()->resolution();
            auto num = res.x * res.y;
            _accumulate_buffer = _device->allocate_buffer<float4>(num);
            camera()->film()->set_accumulate_buffer_view(_accumulate_buffer.view());

            _frame_buffer.reset(_device, num);
            _frame_buffer.synchronize_to_gpu();
            camera()->film()->set_frame_buffer_view(_frame_buffer.device_buffer_view());
        }

        void CUDATask::update() {
            _integrator->update();
        }

        void CUDATask::render_gui(double dt) {
            _dt = dt;
            Clock clock;
            _integrator->render();
            cout << clock.elapse_s() << endl;
        }

        FrameBufferType *CUDATask::download_frame_buffer() {
            _frame_buffer.synchronize_to_cpu();
            return _frame_buffer.data();
        }
    }
}