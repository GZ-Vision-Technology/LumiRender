//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/concepts.h"
#include "render/include/shape.h"
#include "render/include/accelerator.h"
#include "core/backend/managed.h"
#include <embree3/rtcore.h>

namespace luminous {
    inline namespace cpu {

        class EmbreeAccel : public Accelerator {
        private:
            static RTCDevice _rtc_device;
            RTCScene _rtc_scene{};
        public:
            EmbreeAccel(const Scene *scene);

            ~EmbreeAccel();

            static void init_device();

            NDSC RTCScene rtc_scene() const { return _rtc_scene; }

            static RTCDevice rtc_device() { return _rtc_device; }

            void clear() override {}

            NDSC uint64_t handle() const override { return reinterpret_cast<uint64_t>(_rtc_scene); }

            NDSC std::string description() const override;

            RTCScene build_mesh(const Managed<float3> &positions,
                                const Managed<TriangleHandle> &triangles,
                                const MeshHandle &mesh);

            void build_bvh(const Managed<float3> &positions, const Managed<TriangleHandle> &triangles,
                           const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                           const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) override;

        };

    } // luminous::cpu
} // luminous