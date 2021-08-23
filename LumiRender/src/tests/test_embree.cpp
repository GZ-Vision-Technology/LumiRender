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
#define EMBREE_CHECK(expr)                                                                                             \
[&] {                                                                                                              \
expr;                                                                                                          \
auto err = rtcGetDeviceError(rtc_device);                                                                          \
if (err != RTC_ERROR_NONE) {                                                                                   \
printf("wocao !!!\n");                                                                    \
}                                                                                                              \
}()

int main() {
    auto rtc_device = rtcNewDevice(nullptr);
    float3 p[] = {
            make_float3(0.f, 0.f, 0.f),
            make_float3(1.f, 0.f, 0.f),
            make_float3(0.f, 1.f, 0.f),
            make_float3(1.f, 1.f, 0.f)
    };
    TriangleHandle tri[] = {{0, 1, 2},
                            {3, 1, 2}};
    RTCScene rtc_scene = rtcNewScene(rtc_device);
    Transform transform = Transform::translation(make_float3(-1, 0.0, 0));
    Transform transform2 = Transform::translation(make_float3(0, 0.0, 0));

    RTCGeometry rtc_geometry = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    EMBREE_CHECK(rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_VERTEX,
                                            0, RTC_FORMAT_FLOAT3,
                                            p, 0, sizeof(float3), 4));

    EMBREE_CHECK(rtcSetSharedGeometryBuffer(rtc_geometry, RTC_BUFFER_TYPE_INDEX,
                                            0, RTC_FORMAT_UINT3, &tri,
                                            0, sizeof(TriangleHandle), 2));

    EMBREE_CHECK(rtcCommitGeometry(rtc_geometry));
    EMBREE_CHECK(rtcAttachGeometry(rtc_scene, rtc_geometry));
    EMBREE_CHECK(rtcCommitScene(rtc_scene));

    RTCScene main_scene = rtcNewScene(rtc_device);

    RTCGeometry instance = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_INSTANCE);
    EMBREE_CHECK(rtcSetGeometryInstancedScene(instance, rtc_scene));
    EMBREE_CHECK(rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform.mat4x4_ptr()));
    EMBREE_CHECK(rtcCommitGeometry(instance));
    EMBREE_CHECK(rtcAttachGeometry(main_scene, instance));

    RTCGeometry instance2 = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_INSTANCE);
    EMBREE_CHECK(rtcSetGeometryInstancedScene(instance2, rtc_scene));
    EMBREE_CHECK(rtcSetGeometryTransform(instance2, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform2.mat4x4_ptr()));
    EMBREE_CHECK(rtcCommitGeometry(instance2));
    EMBREE_CHECK(rtcAttachGeometry(main_scene, instance2));


    EMBREE_CHECK(rtcCommitScene(main_scene));

    RTCBounds rb;
    rtcGetSceneBounds(main_scene, &rb);

    PerRayData prd;

    Ray ray(make_float3(-0.1, 0.5, 1), make_float3(0, 0, -1));
    intersect_closest((uint64_t) main_scene, ray, &prd);
    prd.hit_point.print();

    ray = Ray(make_float3(-0.9, 0.5, 1), make_float3(0, 0, -1));
    intersect_closest((uint64_t) main_scene, ray, &prd);
    prd.hit_point.print();

    ray = Ray(make_float3(0.1, 0.5, 1), make_float3(0, 0, -1));
    intersect_closest((uint64_t) main_scene, ray, &prd);
    prd.hit_point.print();

    ray = Ray(make_float3(0.9, 0.5, 1), make_float3(0, 0, -1));
    intersect_closest((uint64_t) main_scene, ray, &prd);
    prd.hit_point.print();

    return 0;
}