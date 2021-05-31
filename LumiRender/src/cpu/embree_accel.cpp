//
// Created by Zero on 2021/5/16.
//

#include "embree_accel.h"
#include "cpu_scene.h"

namespace luminous {
    inline namespace cpu {
        RTCDevice EmbreeAccel::_rtc_device = nullptr;

        void EmbreeAccel::init_device() {
            _rtc_device = rtcNewDevice(nullptr);
        }

        EmbreeAccel::EmbreeAccel(const CPUScene *cpu_scene) {
            init_device();
            _rtc_scene = rtcNewScene(_rtc_device);
        }

        std::string EmbreeAccel::description() const {
            return nullptr;
        }

        void EmbreeAccel::build_bvh(const Managed<float3> &positions,
                                    const Managed<TriangleHandle> &triangles,
                                    const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                                    const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) {

        }

        RTCGeometry EmbreeAccel::build_mesh(const Managed<float3> &positions,
                                            const Managed<TriangleHandle> &triangles,
                                            const MeshHandle &mesh) {
            RTCGeometry rtc_geometry = rtcNewGeometry(rtc_device(), RTC_GEOMETRY_TYPE_TRIANGLE);
            auto p_vert = positions.const_host_buffer_view(mesh.vertex_offset, mesh.vertex_count).cbegin();
            rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_VERTEX,
                                       0, RTC_FORMAT_FLOAT3,
                                       p_vert, 0, sizeof(float3), positions.size());
            auto p_tri = triangles.const_host_buffer_view(mesh.triangle_offset, mesh.triangle_count).cbegin();
            rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_INDEX,
                                       0, RTC_FORMAT_UINT3, p_tri,
                                       0, sizeof(TriangleHandle), triangles.size());
            return rtc_geometry;
        }

    } // luminous::cpu
} // luminous