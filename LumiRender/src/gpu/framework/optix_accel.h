//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "cuda_device.h"
#include "graphics/geometry/common.h"
#include "render/include/scene_graph.h"

namespace luminous {
    inline namespace gpu {
        class OptixAccel : public Noncopyable {
        private:
            std::shared_ptr<Device> _device;
            Dispatcher _dispatcher;
            OptixDeviceContext _optix_context;
            uint32_t geom_flags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            size_t _gpu_bvh_bytes = 0;
            OptixTraversableHandle _root_traversable;
        private:
            void create_module();

            OptixTraversableHandle build_bvh(const std::vector<OptixBuildInput> &build_inputs);

        public:
            OptixAccel(std::shared_ptr<Device> device);

            void build(SP<SceneGraph> graph);
        };
    }
}