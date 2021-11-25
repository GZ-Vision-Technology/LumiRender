//
// Created by Zero on 23/11/2021.
//

#include <iostream>
#include "render/filters/filter_base.h"
#include "base_libs/sampling/sampling.h"
#include "base_libs/sampling/distribution.h"

using namespace std;

using namespace luminous;

int main() {

    BufferView<const float> bfv;

    DistribData d(bfv, bfv, 1);

    Distribution distribution(d);

    Array<float, 1> func;
    Array<float, 2> CDF;
    CDistribData<1> cdd(func, CDF, 1);
    Distribution d2(cdd);

    Distribution d3(bfv, bfv, 1);

    return 0;
}