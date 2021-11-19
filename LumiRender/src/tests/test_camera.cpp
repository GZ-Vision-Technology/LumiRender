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
    auto pitch_t = Transform::rotation_x(-tc.pitch);

    cout << pitch_t.mat4x4().to_string() << endl;
    cout << yaw_t.mat4x4().to_string() << endl;
    auto tt = Transform::translation(tc.position);
    return pitch_t * yaw_t;
}

class Camera {
public:
    float yaw{}, pitch{};

    explicit Camera(float4x4 m) {
        update(m);
    }

    void update(const float4x4 &m) {

        float3 x = make_float3(m[0]);
        float3 y = make_float3(m[1]);
        float3 z = make_float3(m[2]);

        float sy = sqrt(sqr(z.y) + sqr(z.z));

        float l_xz = sqrt(sqr(z.x) + sqr(z.z));

        pitch = degrees(std::atan2(z.y, abs(z.z)));
        yaw = degrees(-std::atan2(-z.x, sy));
    }

    Transform camera_to_world_rotation() const {
        auto horizontal = Transform::rotation_y(yaw);
        auto vertical = Transform::rotation_x(-pitch);
        return horizontal * vertical;
    }
};

void test_sensor() {

    TransformConfig tc;
    tc.set_type("yaw_pitch");
    tc.yaw = 30;
    tc.pitch = 20;
    tc.position = make_float3(0, 0, 0);

    auto t1 = create(tc);

    Camera camera(t1.mat4x4());

    cout << t1.mat4x4().to_string() << endl;
    return;

    auto m2 = camera.camera_to_world_rotation();

    auto ea = matrix_to_Euler_angle(t1.mat4x4());

    cout << ea.to_string() << endl;

    cout << m2.mat4x4().to_string() << endl;

    return;

}

int main() {
    test_sensor();
//    test_sampler();
}