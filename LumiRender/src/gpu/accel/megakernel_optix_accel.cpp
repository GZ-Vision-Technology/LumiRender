//
// Created by Zero on 2021/1/10.
//

#include "megakernel_optix_accel.h"
#include <iosfwd>

extern "C" char megakernel_pt[];

namespace luminous {
    inline namespace gpu {

        MegakernelOptixAccel::MegakernelOptixAccel(Device *device, Context *context, const Scene *scene)
                : OptixAccel(device, context, scene) {

            const char *callables[] = {/* "__direct_callable__get_bsdf_0",
                                     "__direct_callable__get_bsdf_1",
                                     "__direct_callable__get_bsdf_2",
                                     "__direct_callable__sample_f_0",
                                     "__direct_callable__sample_f_1", */
                    nullptr};
            ProgramName megakernel_shader{
                    "__raygen__rg",
                    "__closesthit__closest",
                    "__closesthit__any",
                    callables};

            _shader_wrapper = create_shader_wrapper(megakernel_pt, megakernel_shader);

            build_pipeline(_shader_wrapper.program_groups());
        }

        void MegakernelOptixAccel::launch(uint2 res, Managed<LaunchParams> &launch_params) {
            auto stream = dynamic_cast<CUDADispatcher *>(_dispatcher.impl_mut())->stream;
            auto x = res.x;
            auto y = res.y;
            launch_params->traversable_handle = _root_as_handle;
            launch_params.synchronize_to_device();
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
            _shader_wrapper.clear();
            OptixAccel::clear();
        }

        MegakernelOptixAccel::~MegakernelOptixAccel() {
            clear();
        }
    }
}