//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/sensors/sensor_handle.h"

using namespace luminous;
using namespace std;

FilmHandle create_film() {
    FilmConfig fc;
    fc.type = "RGBFilm";
    fc.resolution = make_int2(500);
    return FilmHandle::create(fc);
}

void test_sensor() {
    SensorConfig config;
    config.type = "PinholeCamera";
    config.fov_y = 90;
    config.velocity = 20;

    TransformConfig tc;
    tc.type = "yaw_pitch";

    tc.yaw = -180;
    tc.pitch = 0;
    tc.position = make_float3(2,3,5);

    tc.mat4x4 = make_float4x4(1);
    config.transform_config = tc;

    auto camera = SensorHandle::create(config);

    auto film = create_film();

    camera.set_film(film);

    SensorSample ss;
    ss.p_film = make_float2(250,250);

    Ray ray;
    camera.generate_ray(ss, &ray);

    cout << camera.to_string() << endl;
    cout << ray.to_string() << endl;

    cout << camera.forward().to_string() << endl;

    auto mat = camera.camera_to_world_rotation().mat4x4();

    tc.type = "matrix4x4";
    tc.mat4x4 = mat;

    config.transform_config = tc;
    auto c2 = SensorHandle::create(config);

    cout << camera.to_string() << endl;

    auto f = camera.film();
    cout << f->to_string() << endl;
    f->set_resolution(make_int2(300));
    auto f2 = camera.film();
    cout << f2->to_string();
}

int main() {
    test_sensor();
//    test_sampler();
}