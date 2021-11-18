//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/sensors/sensor.h"
#include "render/sensors/shader_include.h"

using namespace luminous;
using namespace std;

LM_NODISCARD Transform create(TransformConfig tc) {
        auto yaw_t = Transform::rotation_y(tc.yaw);
        auto pitch_t = Transform::rotation_x(tc.pitch);
        auto tt = Transform::translation(tc.position);
        return tt * pitch_t * yaw_t;
}

class Camera {
public:
    float yaw{}, pitch{};
    explicit Camera(float4x4 m) {
        update(m);
    }

    void update(const float4x4 &m) {
        float sy = sqrt(sqr(m[2][1]) + sqr(m[2][2]));
        pitch = degrees(std::atan2(-m[2][1], m[2][2]));
        yaw = degrees(-std::atan2(-m[2][0], sy));
    }

    Transform camera_to_world_rotation() const {
        auto horizontal = Transform::rotation_y(yaw);
        auto vertical = Transform::rotation_x(pitch);
        return horizontal *vertical;
    }
};

void test_sensor() {

    TransformConfig tc;
    tc.set_type("yaw_pitch");
    tc.yaw = 95;
    tc.pitch = 20;
    tc.position = make_float3(0,0,0);

    auto t1 = create(tc);

    Camera camera(t1.mat4x4());

    cout << t1.mat4x4().to_string() << endl;

    auto m2 = camera.camera_to_world_rotation();

    cout << m2.mat4x4().to_string() << endl;

    return;

}

int main() {
    test_sensor();
//    test_sampler();
}