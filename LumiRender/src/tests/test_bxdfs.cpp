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
#include "render/scattering/bsdf_wrapper.cpp"
#include "render/scattering/bsdf_data.h"

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

float sample_bsdf(BSDFWrapper bsdf_wrapper, luminous::float3 wo, float eta) {
    PCGSampler sampler{1};
    int num = 100000;
    Spectrum ret{0.f};
    auto wi = wo;
    bool valid = refract(wo, luminous::make_float3(0,0,1), eta, &wi);
    if (!valid) {
        return 0;
    }
    for (int i = 0; i < num; ++i) {
        auto bs = bsdf_wrapper.sample_f(wo, sampler.next_1d(), sampler.next_2d());
        if (bs.PDF == 0) {
            --i;
            continue;
        }
        auto result = bs.f_val / bs.PDF;
        ret += result / float(num);
    }
    cout << ret.to_string() << endl;
    return ret.x;
}

void test_microfacet_transmission(float deg) {
    PCGSampler sampler{1};
    cout << "\ndeg = " << deg << endl;
    auto wo = luminous::spherical_direction(radians(deg), 120);
    auto n = luminous::make_float3(0, 0, 1);
    auto s = luminous::make_float3(1, 0, 0);
    float eta = 1.5;
    cout << "specular ";
    BSDFWrapper bsdf_wrapper{n, n, s, BSDF{create_glass_bsdf_test(luminous::make_float4(1.f), eta, false, true)}};
    float rs =  sample_bsdf(bsdf_wrapper, wo, eta);
    cout << "rough ";
    BSDFWrapper bsdf_wrapper2{n, n, s,
                              BSDF{create_rough_glass_bsdf_test(luminous::make_float4(1.f), eta, 0.002, 0.002, false,
                                                                true)}};
    float rr = sample_bsdf(bsdf_wrapper2, wo ,eta);
    cout << "rr : rs = " << rr / rs;
    cout << endl;
}

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

    cout << sizeof (luminous::float3) << endl;

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