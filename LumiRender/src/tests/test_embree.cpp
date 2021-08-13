//
// Created by Zero on 2021/8/12.
//

#include <embree3/rtcore.h>
#include "base_libs/common.h"
#include "render/include/interaction.h"
#include "render/include/trace.h"
#include "util/clock.h"

using namespace std;

using namespace luminous;

int main() {
    auto rtc_device = rtcNewDevice(nullptr);
    float3 p[] = {
            make_float3(0.f, 0.f, 0.f),
            make_float3(1.f, 0.f, 0.f),
            make_float3(0.f, 1.f, 0.f),
            make_float3(1.f, 1.f, 0.f)
    };
    TriangleHandle tri[] = {{0,1,2},{3,1,2}};
    RTCScene rtc_scene = rtcNewScene(rtc_device);
    Transform transform = Transform::translation(make_float3(-1,0.0,0));
    Transform transform2 = Transform::translation(make_float3(0,0.0,0));

    RTCGeometry rtc_geometry = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_VERTEX,
                               0, RTC_FORMAT_FLOAT3,
                               p, 0, sizeof(float3), 4);

    rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_INDEX,
                               0, RTC_FORMAT_UINT3, &tri,
                               0, sizeof(TriangleHandle), 2);

    rtcCommitGeometry(rtc_geometry);

    rtcAttachGeometry(rtc_scene, rtc_geometry);
    rtcCommitScene(rtc_scene);

    RTCGeometry instance = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_INSTANCE);
    RTCGeometry instance2 = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(instance, rtc_scene);
    rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform.mat4x4_ptr());

    rtcSetGeometryInstancedScene(instance2, rtc_scene);
    rtcSetGeometryTransform(instance2, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform2.mat4x4_ptr());

    rtcCommitGeometry(instance);
    rtcCommitGeometry(instance2);

    RTCScene main_scene = rtcNewScene(rtc_device);
    rtcAttachGeometry(main_scene, instance);
//    rtcAttachGeometry(main_scene, instance2);


    rtcCommitScene(main_scene);

    Ray ray(make_float3(-0.9, 0.3, 1), make_float3(0,0,-1));
    PerRayData prd;

    Clock clk;
    intersect_closest((uint64_t)main_scene, ray, &prd);

    RTCBounds rb;
    rtcGetSceneBounds(main_scene, &rb);

    auto t = clk.elapse_ms();
    prd.closest_hit.print();

    return 0;
}