//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/sensors/sensor.h"

using namespace luminous;
using namespace std;

void test_sensor() {
    SensorConfig config;
    config.type = "PinholeCamera";
    config.fov_y = 90;
    config.velocity = 20;

    TransformConfig tc;
    tc.type = "yaw_pitch";

    tc.yaw = 0;
    tc.pitch = 0;
    tc.position = make_float3(0,0,0);
    config.film_config.type = "RGBFilm";
    config.film_config.resolution = make_uint2(500,500);
    tc.mat4x4 = make_float4x4(1);
    config.transform_config = tc;

    auto camera = Sensor::create(config);

    SensorSample ss;
    ss.p_film = make_float2(250,250);

    auto mat = camera.camera_to_world_rotation().mat4x4();

    tc.type = "matrix4x4";
    tc.mat4x4 = mat;


//    camera.move(camera.forward());

    Ray ray;
    camera.generate_ray(ss, &ray);
    cout << camera.to_string() << endl;

    cout << ray.to_string() << endl;

//    camera.update_fov_y(5);
    camera.set_yaw(90);
    camera.set_pitch(45);
    camera.generate_ray(ss, &ray);

    cout << ray.to_string() << endl;

}

int main() {
    test_sensor();
//    test_sampler();
}