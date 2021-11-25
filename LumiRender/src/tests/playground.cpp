//
// Created by Zero on 23/11/2021.
//

#include <iostream>
#include "render/filters/filter_base.h"
#include "base_libs/sampling/sampling.h"
#include "base_libs/sampling/distribution.h"

using namespace std;

using namespace luminous;

Distribution2D create(const float *func, int U, int V) {
    auto builder2d = Distribution2D::create_builder(func, U, V);
    std::vector<Distribution1D> conditional_v;
    for (int i = 0; i < builder2d.conditional_v.size(); ++i) {
        auto &builder = builder2d.conditional_v[i];
        Distribution1D distribution(BufferView<const float>{builder.func.data(), builder.func.size()},
                                    BufferView<const float>{builder.CDF.data(), builder.CDF.size()},
                                    builder.func_integral);
        conditional_v.push_back(distribution);
    }
    Distribution1D marginal(BufferView<const float>(builder2d.marginal.func.data(), builder2d.marginal.func.size()),
                            BufferView<const float>(builder2d.marginal.CDF.data(), builder2d.marginal.CDF.size()),
                            builder2d.marginal.func_integral);
    Distribution2D ret(BufferView<const Distribution1D>{conditional_v.data(), conditional_v.size()}, marginal);
    float PDF;
    float2 p = ret.sample_continuous(make_float2(0.5f), &PDF);

    return ret;
}

int main() {

    BufferView<const float> bfv;

    DistribData d(bfv, bfv, 1);

    TDistribution distribution(d);

    Array<float, 1> func;
    Array<float, 2> CDF;
    CDistribData<1> cdd(func, CDF, 1);
    TDistribution d2(cdd);

    TDistribution d3(bfv, bfv, 1);
    vector<float> ret;
    for (int i = 0; i < 100; ++i) {
        ret.push_back(i);
    }

    auto sd = create_static_distrib2d<10, 10>(ret.data());

    auto db = Distribution2D::create_builder(func.data(), 10, 10);

    float PDF;
    float2 p = sd.sample_continuous(make_float2(0.5f), &PDF);

    create(ret.data(), 10, 10);



    return 0;
}