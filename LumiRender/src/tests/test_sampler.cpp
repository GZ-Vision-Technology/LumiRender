//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/include/sampler.h"
#include "render/include/sensor.h"

using namespace luminous;
using namespace std;

void test_sampler() {
    auto config = SamplerConfig();
    config.type = "LCGSampler";
    config.spp = 9;
    auto sampler = SamplerHandle::create(config);
    cout << sampler.to_string() << endl;
}

void test_sensor() {
    SensorConfig config;
    config.type = "PinholeCamera";
    config.fov_y = 30;
    config.velocity = 20;

    TransformConfig tc;
    tc.type = "yaw_pitch";

    tc.yaw = 20;
    tc.pitch = 30;
    tc.position = make_float3(2,3,5);

    tc.mat4x4 = make_float4x4(1);
    config.transform_config = tc;

    auto camera = SensorHandle::create(config);

//    cout << camera.to_string();
}

int main() {

    test_sensor();

}