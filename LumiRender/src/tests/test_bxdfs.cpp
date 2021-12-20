//
// Created by Zero on 11/12/2021.
//

#include "render/scattering/bsdf_ty.h"
#include "iostream"
#include "render/scattering/specular_scatter.h"
#include "render/scattering/microfacet.h"
#include "render/scattering/bsdfs.h"
#include "render/scattering/fresnel.h"
#include "render/scattering/diffuse_scatter.h"
#include "render/samplers/independent.cpp"

using namespace luminous;
using namespace std;

void test_microfacet() {
    PCGSampler sampler{1};

    Microfacet<GGX> microfacet{0.1};

    float3 wo = make_float3(-0.889549971, -0.295042098, 0.348785102);
    float3 wi = make_float3(0.558969319, 0.284566134, -0.778829575);

    auto bsdf = create_rough_glass_bsdf(make_float4(1.), 1.5, 0.1, 0.1);

//    auto ret = bsdf.eval(wo, wi);

    auto pdf = bsdf.PDF(wo, wi);

    int i = 0;
}

int main() {

    test_microfacet();

    return 0;
}