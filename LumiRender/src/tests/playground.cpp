//
// Created by Zero on 23/11/2021.
//

#include "render/include/distribution_mgr.h"
#include "core/backend/managed.h"
#include "base_libs/sampling/alias_table.h"
#include "base_libs/sampling/distribution.h"
#include "render/samplers/independent.cpp"
//#include "render/include/distribution_mgr.cpp"
//#include "cpu/cpu_impl.cpp"
#include "util/clock.h"

using namespace luminous;
using namespace std;


void test_2d() {
//    auto device = create_cpu_device();
//    DistributionMgr _distribution_mgr{device.get()};
    vector<float> weight;
    static constexpr int width = 5;
    static constexpr int height = 2 * width;
    static constexpr int area = width * height;
    int a[area] = {0};
    for(int i = 0; i < area; ++i) {
        weight.push_back(i);
        a[i] = 0;
    }

    auto d2d = create_static_distrib2d<width, height>(weight.data());
    auto builder = Distribution2D::create_builder(weight.data(), width, height);

    Clock clk;

    PCGSampler sampler;
    int num = 1000000;
    for (int i = 0; i < num; ++i) {
        float p;
        int2 offset;
        auto ret = d2d.sample_continuous(sampler.next_2d(), &p, &offset);
        int index = offset.y * width + offset.x;
        a[index] += 1;
    }
    int cc = 0;
    for (int i = 0; i < area; ++i) {
//        cc += a[i];
        cout << i << "  " << a[i] << "  " << a[i] / float(i) << endl;
    }
//    cout << cc << endl;
//
//    cout << clk.get_elapsed_time() << endl;
}

int main() {

    test_2d();

    return 0;
}