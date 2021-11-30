//
// Created by Zero on 23/11/2021.
//

#include <iostream>
#include "render/filters/filter_base.h"
#include "base_libs/sampling/sampling.h"
#include "base_libs/sampling/distribution.h"
#include "base_libs/optics/optics.h"

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
    auto cos_theta_i = 0.5f;
    auto r = fresnel_dielectric(cos_theta_i, 1, 1.5);

    auto r2 = fresnel_dielectric(cos_theta_i, 1.5);

    float etai = 1;
    float etat = 1.5;
    float k = 0.9;
    auto rc = fresnel_conductor(cos_theta_i, etai, etat, k);
    auto rc2 = fresnel_complex(cos_theta_i, Complex<float>(etat, k));
    return 0;
}