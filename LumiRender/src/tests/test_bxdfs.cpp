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

using namespace luminous;
using namespace std;


int main() {

    float3 wo = make_float3(0.12419761, -0.462170839, -0.878050804);
    float3 wi = make_float3(-0.186296403, 0.693256259, -0.696196556);
    float3 wt{};
    float eta = 0.66666;
    float3 n = make_float3(0,0,1);

    bool valid = refract(wo, -n, eta, &wt);

    cout << wt.to_string().c_str() << endl;

    cout << Frame::sin_theta(wt) << endl;
    cout << Frame::sin_theta(wo) << endl;

    cout << Frame::sin_theta(wt) /Frame::sin_theta(wo)<< endl;


//    auto fresnel = FresnelDielectric(1.5);
//
//    for (int i = 0; i < 90; ++i) {
//        float r = radians(float(i));
//        float cos_theta = std::cos(r);
//        float fr = fresnel.eval(-cos_theta);
//        cout << "cos :" << cos_theta << " deg:" << i << " fr :" <<fr << endl;
//    }


//    DiffuseBSDF bsdf(Spectrum{1.f}, MicrofacetNone{}, FresnelNoOp{}, DiffuseReflection{});
//
//    float3 wi = make_float3(0,1,0);
//    float3 wo = make_float3(0,1,0);
//    Spectrum color{1.f};

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