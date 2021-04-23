//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/samplers/sampler.h"
#include "render/sensors/sensor.h"
using namespace luminous;
using namespace std;

template<typename T>
class Test {

};

void test_sampler() {
//    TASK_TAG("test_sampler");
    auto config = SamplerConfig();
    cout << string(typeid(PCGSampler).name()).c_str();
    config.set_full_type("PCGSampler");
    config.spp = 9;
    auto sampler = Sampler::create(config);
    cout << sampler.to_string() << endl;
    cout << sampler.next_2d().to_string() << endl;
    auto s2 = sampler;
    auto f = 0.f;
    for (int i = 0; i < 10000000; ++i) {
        f = f + sampler.next_1d();
    }
    cout << sampler.to_string();
}

void test_sensor() {
    SensorConfig config;
    config.set_full_type("PinholeCamera");
    config.fov_y = 30;
    config.velocity = 20;

    TransformConfig tc;
    tc.set_type("yaw_pitch");

    tc.yaw = 20;
    tc.pitch = 15.6;
    tc.position = make_float3(2,3,5);


    tc.mat4x4 = make_float4x4(1);
    config.transform_config = tc;
    config.film_config.set_full_type("RGBFilm");

    auto camera = Sensor::create(config);

    cout << camera.to_string();
}

int main() {
    test_sensor();
}