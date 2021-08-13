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

        EmbreeAccel::EmbreeAccel() {
            init_device();
            _rtc_scene = rtcNewScene(_rtc_device);
//            rtcSetSceneBuildQuality(_rtc_scene, RTC_BUILD_QUALITY_HIGH);
//            rtcSetSceneFlags(_rtc_scene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
        }

        EmbreeAccel::~EmbreeAccel() {
            rtcReleaseScene(_rtc_scene);
        }

        std::string EmbreeAccel::description() const {
            return string_printf("EmbreeAccel");
        }

        void EmbreeAccel::build_bvh(const Managed<float3> &positions,
                                    const Managed<TriangleHandle> &triangles,
                                    const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                                    const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) {
            TASK_TAG("build embree bvh");
            std::vector<RTCScene> mesh_scenes;
            mesh_scenes.reserve(meshes.size());
            for (const auto &mesh : meshes) {
                RTCScene mesh_scene = build_mesh(positions, triangles, mesh);
                mesh_scenes.push_back(mesh_scene);
            }

            for (int i = 0; i < instance_list.size(); ++i) {
                uint mesh_idx = instance_list[i];
                RTCScene mesh_scene = mesh_scenes[mesh_idx];
                RTCGeometry instance = rtcNewGeometry(rtc_device(), RTC_GEOMETRY_TYPE_INSTANCE);
                rtcSetGeometryInstancedScene(instance, mesh_scene);
                rtcSetGeometryTimeStepCount(instance,1);
                uint transform_idx = inst_to_transform[i];
                const Transform &transform = transform_list[transform_idx];
                rtcAttachGeometry(_rtc_scene, instance);
                rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform.mat4x4_ptr());
                rtcCommitGeometry(instance);
                rtcReleaseGeometry(instance);
            }
            rtcCommitScene(_rtc_scene);
        }

        RTCScene EmbreeAccel::build_mesh(const Managed<float3> &positions,
                                            const Managed<TriangleHandle> &triangles,
                                            const MeshHandle &mesh) {
            RTCScene scene = rtcNewScene(rtc_device());
            RTCGeometry rtc_geometry = rtcNewGeometry(rtc_device(), RTC_GEOMETRY_TYPE_TRIANGLE);
            auto pos = positions.const_host_buffer_view(mesh.vertex_offset, mesh.vertex_count);
            rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_VERTEX,
                                       0, RTC_FORMAT_FLOAT3,
                                       pos.cbegin(), 0, sizeof(float3), pos.size());
            auto tri_list = triangles.const_host_buffer_view(mesh.triangle_offset, mesh.triangle_count);
            rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_INDEX,
                                       0, RTC_FORMAT_UINT3, tri_list.cbegin(),
                                       0, sizeof(TriangleHandle), tri_list.size());
            rtcCommitGeometry(rtc_geometry);
            rtcAttachGeometry(scene, rtc_geometry);
            rtcReleaseGeometry(rtc_geometry);
            rtcCommitScene(scene);
            return scene;
        }

    } // luminous::cpu
} // luminous