//
// Created by Zero on 11/12/2021.
//

#include "render/scattering/bsdf.h"
#include "iostream"
#include "render/scattering/specular_scatter.h"
#include "render/scattering/microfacet_scatter.h"
#include "render/scattering/diffuse_scatter.h"

struct TestA {

};

struct TestB {

};

using zt = std::tuple<TestB,TestA>;

using namespace luminous;
using namespace std;
int main() {


//    using Specular = BxDFs<SpecularReflection, SpecularTransmission>;

cout << sizeof(zt) << endl;

    return 0;
}