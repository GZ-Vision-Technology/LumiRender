//
// Created by Zero on 11/12/2021.
//

#include "render/scattering/bsdf_ty.h"
#include "iostream"
#include "render/scattering/specular_scatter.cpp"
#include "render/scattering/bsdfs.h"
#include "render/scattering/lambert_scatter.cpp"
#include "render/samplers/independent.cpp"
#include "render/scattering/bsdf_wrapper.cpp"
#include "render/scattering/bsdf_data.h"
#include "render/scattering/disney_bsdf.cpp"
#include "render/scattering/microfacet_scatter.cpp"

using namespace luminous;
using namespace std;

//void test_microfacet() {
//    PCGSampler sampler{1};
//
//    Microfacet<GGX> microfacet{0.1};
//
//    float3 wo = make_float3(-0.889549971, -0.295042098, 0.348785102);
//    float3 wi = make_float3(0.558969319, 0.284566134, -0.778829575);
//
//    auto bsdf = create_rough_glass_bsdf(make_float4(1.), 1.5, 0.002, 0.002, false, true);
////    sampler.next_2d();
//    auto u = sampler.next_2d();
//    auto ret = bsdf.sample_f(wo, 0.9, u);
//
//    cout << u.to_string() << endl;
//    cout << wo.to_string() << endl;
//
//}

//void test_refract() {
//    float3 wo = spherical_direction(radians(60), 0);
//    float eta = 1.5;
//    float3 wi{};
//    float3 n = make_float3(0, 0, 1);
//    bool valid = refract(wo, n, eta, &wi);
//    float fr = fresnel_dielectric(Frame::abs_cos_theta(wo), eta);
//
//    auto bsdf = create_glass_bsdf_test(make_float4(1.f), 1.5, false, true);
//
//    auto bs = bsdf.sample_f(wo, 0, make_float2(0));
//
//    cout << wi.to_string() << endl;
//    cout << wo.to_string() << endl;
//    cout << fr << endl;
//    fr = fresnel_dielectric(Frame::abs_cos_theta(wi), 1 / eta);
//    cout << fr << endl;
//}


enum FresnelType : uint8_t {
    NoOp,
    Dielectric,
    Conductor
};

void test_bsdf_data() {
    cout << sizeof (BxDFFlags) << endl;
    cout << int(BxDFFlags::SpecRefl);
}

int main() {

    cout << sizeof (luminous::BSDF)<< endl;

    disney::DiffuseTransmission dt;

    BSDFHelper helper;
    auto ret = dt.eval(luminous::make_float3(0,0,1), luminous::make_float3(0,0,-1), helper);
    cout << ret.to_string() << endl;
//    test_bsdf_data();

//    BSDFData bsdf_data = BSDFData::create_diffuse_data(make_float4(12));
//
//    cout << bsdf_data.diffuse_data.color.to_string() << endl;
//    cout << sizeof(BSDF) << endl;

//    test_refract();
//    cout << endl;
//    test_microfacet();
//    for (int i = 1; i < 90; i += 3) {
//        test_microfacet_transmission(i);
////        break;
//    }
    cout << endl;
    return 0;
}