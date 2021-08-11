//
// Created by Zero on 2021/5/16.
//

#include "embree_accel.h"
#include "util/stats.h"

namespace luminous {
    inline namespace cpu {
        RTCDevice EmbreeAccel::_rtc_device = nullptr;

        void EmbreeAccel::init_device() {
            _rtc_device = rtcNewDevice(nullptr);
        }

        EmbreeAccel::EmbreeAccel(const CPUScene *cpu_scene) {
            init_device();
            _rtc_scene = rtcNewScene(_rtc_device);
            rtcSetSceneBuildQuality(_rtc_scene, RTC_BUILD_QUALITY_HIGH);
            rtcSetSceneFlags(_rtc_scene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
            _scene_data = cpu_scene->_scene_data.get();
        }

        std::string EmbreeAccel::description() const {
            return string_printf("EmbreeAccel");
        }

        void EmbreeAccel::build_bvh(const Managed<float3> &positions,
                                    const Managed<TriangleHandle> &triangles,
                                    const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                                    const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) {
            TASK_TAG("build embree bvh");
            std::vector<RTCGeometry> rtc_geometries;
            rtc_geometries.reserve(meshes.size());
            for (const auto &mesh : meshes) {
                RTCGeometry rtc_geometry = build_mesh(positions, triangles, mesh);
                rtc_geometries.push_back(rtc_geometry);
            }
            std::vector<RTCGeometry> rtc_instances;
            rtc_instances.reserve(instance_list.size());
            for (int i = 0; i < instance_list.size(); ++i) {
                uint mesh_idx = instance_list[i];
                RTCScene rtc_scene{nullptr};
                RTCGeometry mesh_geometry = rtc_geometries[mesh_idx];
                rtcCommitGeometry(mesh_geometry);
                rtcAttachGeometry(rtc_scene, mesh_geometry);
                RTCGeometry instance = rtcNewGeometry(rtc_device(), RTC_GEOMETRY_TYPE_INSTANCE);
                rtcSetGeometryInstancedScene(instance, rtc_scene);
                rtc_instances.push_back(instance);
                rtcAttachGeometry(_rtc_scene, instance);
                rtcReleaseGeometry(instance);
            }
            for (auto geometry : rtc_geometries) {
                rtcReleaseGeometry(geometry);
            }
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