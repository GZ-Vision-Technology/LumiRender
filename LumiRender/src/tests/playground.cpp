//
// Created by Zero on 23/11/2021.
//

#include <iostream>
#include "render/filters/filter_base.h"

using namespace std;

using namespace luminous;

int main() {

    constexpr int size = FilterSampler::tab_size;

    float a[sqr(size)] = {};

    for (int i = 0; i < sqr(size); ++i) {
        a[i] = 1.f;
    }

    FilterSampler filter_sampler(a);



    return 0;
}