//
// Created by Zero on 23/11/2021.
//

#include <iostream>
#include "render/scattering/microfacet.cpp"
#include "render/scattering/dielectric.cpp"

using namespace std;

using namespace luminous;

void test_dielectric() {

    auto kr = float4(1.f);
    auto kt = float4(1.f);

    auto eta = 1.5f;
    float r = 0.1;
    DielectricBxDF bxdf(kr, kt, eta, r, r);

    auto wi = spherical_direction(radians(50), 0);
    auto wo = spherical_direction(radians(100), Pi);

    auto l1 = length(wi);
    auto l2 = length(wo);

    auto f = bxdf.eval(wo, wi);

    cout << f.to_string() << endl;

    return;
}

int main() {

    test_dielectric();

    return 0;
}