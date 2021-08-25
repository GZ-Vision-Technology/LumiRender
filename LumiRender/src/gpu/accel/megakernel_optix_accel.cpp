//
// Created by Zero on 2021/1/10.
//

#include "megakernel_optix_accel.h"
#include <optix_function_table_definition.h>
#include "gpu/gpu_scene.h"
#include <iosfwd>

extern "C" char megakernel_pt[];

namespace luminous {
    inline namespace gpu {
        ProgramName program_name{"__raygen__rg",
                                 "__closesthit__closest",
                                 "__closesthit__any",
                                 "__miss__closest",
                                 "__miss__any"};

        MegakernelOptixAccel::MegakernelOptixAccel(const SP<Device> &device, const GPUScene *gpu_scene,
                                                   Context *context)
                : OptixAccel(device, context, gpu_scene),
                _shader_wrapper((create_shader_wrapper(megakernel_pt, program_name))) {
            build_pipeline(_shader_wrapper.program_groups());
        }

        void MegakernelOptixAccel::launch(uint2 res, Managed<LaunchParams> &launch_params) {
            auto stream = dynamic_cast<CUDADispatcher *>(_dispatcher.impl_mut())->stream;
            auto x = res.x;
            auto y = res.y;
            launch_params->traversable_handle = _root_ias_handle;
            launch_params.synchronize_to_gpu();
            OPTIX_CHECK(optixLaunch(_optix_pipeline,
                                    stream,
                                    launch_params.device_ptr<CUdeviceptr>(),
                                    sizeof(LaunchParams),
                                    _shader_wrapper.sbt_ptr(),
                                    x,
                                    y,
                                    1u));

            _dispatcher.wait();
        }

        void MegakernelOptixAccel::clear() {
            optixPipelineDestroy(_optix_pipeline);
            _shader_wrapper.clear();
            OptixAccel::clear();
        }

        MegakernelOptixAccel::~MegakernelOptixAccel() {
            clear();
        }
    }
}