//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/samplers/sampler.h"
#include "render/sensors/sensor.h"
#include "render/integrators/wavefront/work_items.h"
#include "gpu/framework/cuda_impl.h"
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
    auto s2 = sampler;
    auto f = 0.f;
    sampler.start_pixel_sample(make_uint2(0), 0, 0);
    cout << sampler.next_2d().to_string() << endl;
    cout << sampler.next_2d().to_string() << endl;

    sampler.start_pixel_sample(make_uint2(0), 0, 2);
    cout << sampler.next_2d().to_string() << endl;
//    cout << sampler.next_2d().to_string() << endl;
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
    tc.position = luminous::make_float3(2,3,5);


    tc.mat4x4 = make_float4x4(1);
    config.transform_config = tc;
    config.film_config.set_full_type("RGBFilm");

    auto camera = Sensor::create(config);

    cout << camera.to_string();
}


class TT{
public:
    TT() = default;

    TT(TT &&other) {
        cout << "move construct" << endl;
    }

    ~TT() {
        cout << "~TT" << endl;
    }
};

void test_soa() {
    using namespace luminous;
    auto device = luminous::create_cuda_device();

    luminous::SOA<luminous::float4> sf3(9, device.get());


    cout << SOA<luminous::float3>::definitional << endl;
    cout << SOA<luminous::float3*>::definitional << endl;
}

void test_move() {
    vector<TT> vt;
    TT t;
    vt.push_back(std::move(t));
    cout << "wocao" << endl;
}

int main() {
//    test_sampler();
//    test_move();
    test_soa();
}