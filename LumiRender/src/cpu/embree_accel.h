//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "graphics/math/common.h"
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

            void build_bvh(const vector<const float3> &positions, const vector<const TriangleHandle> &triangles,
                           const vector<MeshHandle> &meshes, const vector<uint> &instance_list,
                           const vector<Transform> &transform_list, const vector<uint> &inst_to_transform);
        };

    } // luminous::cpu
} // luminous