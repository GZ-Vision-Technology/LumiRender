//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "base_libs/math/common.h"
#include <embree3/rtcore.h>
#include "core/concepts.h"
#include "render/include/shape.h"
#include "core/backend/managed.h"
#include "cpu_scene.h"

namespace luminous {
    inline namespace cpu {
        class CPUScene;

        class EmbreeAccel : public Noncopyable {
        private:
            static RTCDevice _rtc_device;
            RTCScene _rtc_scene;
            size_t _bvh_size_in_bytes{0u};
        public:
            EmbreeAccel(const CPUScene *cpu_scene);

            static void init_device();

            static RTCDevice rtc_device() { return _rtc_device; }

            size_t bvh_size_in_bytes() const { return _bvh_size_in_bytes; }

            NDSC std::string description() const;

            RTCGeometry build_mesh(const Managed<float3> &positions,
                                   const Managed<TriangleHandle> &triangles,
                                   const MeshHandle &mesh);

            void build_bvh(const Managed<float3> &positions, const Managed<TriangleHandle> &triangles,
                           const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                           const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform);
        };

    } // luminous::cpu
} // luminous