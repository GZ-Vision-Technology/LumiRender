//
// Created by Zero on 11/12/2021.
//

#include "render/scattering/bsdf.h"
#include "iostream"
#include "render/scattering/specular_scatter.h"
#include "render/scattering/microfacet.h"
#include "render/scattering/fresnel.h"
#include "render/scattering/microfacet_scatter.h"
#include "render/scattering/diffuse_scatter.h"

struct TestA {

};

struct TestB {

};

using namespace luminous;
using namespace std;

using zt = std::tuple<TestB,TestA>;

using DiffuseBSDF = BSDF<Spectrum, MicrofacetNone, FresnelNoOp, DiffuseReflection>;
int main() {

    DiffuseBSDF bsdf(Spectrum{1.f}, MicrofacetNone{}, FresnelNoOp{}, DiffuseReflection{});

    float3 wi = make_float3(0,1,0);
    float3 wo = make_float3(0,1,0);
    Spectrum color{1.f};

//    bsdf.eval(wo, wi);

//    bsdf.for_each([&](auto bxdf) {
//        cout << bxdf.match_flags(BxDFFlags::Transmission);
//        auto r = bxdf.eval(wo, wi, color, FresnelNoOp{}, MicrofacetNone{});
//        cout << r.to_string() << endl;
//        return true;
//    });

//    auto ret = bsdf.eval(wo, wi);

//    using Specular = BxDFs<SpecularReflection, SpecularTransmission>;

//    cout << sizeof(zt) << endl;

    return 0;
}