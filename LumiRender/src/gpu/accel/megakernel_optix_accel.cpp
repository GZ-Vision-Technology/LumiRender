//
// Created by Zero on 2021/1/10.
//

#include "megakernel_optix_accel.h"
#include <optix_function_table_definition.h>
#include "gpu/gpu_scene.h"
#include <iosfwd>



namespace luminous {
    inline namespace gpu {



        void MegakernelOptixAccel::launch(uint2 res, Managed<LaunchParams> &launch_params) {
            auto stream = dynamic_cast<CUDADispatcher *>(_dispatcher.impl_mut())->stream;
            auto x = res.x;
            auto y = res.y;
            launch_params->traversable_handle = _root_ias_handle;
            launch_params.synchronize_to_gpu();
            OPTIX_CHECK(optixLaunch(_optix_pipeline2,
                                    stream,
                                    launch_params.device_ptr<CUdeviceptr>(),
                                    sizeof(LaunchParams),
                                    &_sbt,
                                    x,
                                    y,
                                    1u));

            _dispatcher.wait();
        }

        void MegakernelOptixAccel::clear() {
            optixPipelineDestroy(_optix_pipeline2);
            _program_group_table.clear();
            _device_ptr_table = {};
            OptixAccel::clear();
        }

        MegakernelOptixAccel::~MegakernelOptixAccel() {
            clear();
        }
    }
}