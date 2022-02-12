//
// Created by Zero on 23/11/2021.
//

#include "render/include/distribution_mgr.h"
#include "gpu/framework/cuda_impl.h"
#include "core/backend/managed.h"
#include "base_libs/sampling/alias_table.h"
#include "base_libs/sampling/distribution.h"
#include "render/samplers/independent.cpp"

using namespace luminous;
using namespace std;

void test_alias() {
    auto[table, PDF, sum] = luminous::create_alias_table(vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    AliasData alias_data{BufferView(table.data(), table.size()), BufferView(PDF.data(), PDF.size()), 0.f};

    PCGSampler sampler;

    TAliasTable alias_table(alias_data);

    float ur, pdf;
    int ofs;
    float u = 0.1;
    auto uc = alias_table.sample_discrete(u, &pdf, &ur);

    int a[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto builder = DichotomySampler::create_builder({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});



    DichotomySampler distribution(BufferView<const float>(builder.func.data(), builder.func.size()),
                                  BufferView<const float>(builder.CDF.data(), builder.CDF.size()), builder.func_integral);

    int count = 1000000;
    for (int i = 0; i < count; ++i) {
        ofs = alias_table.sample_discrete(sampler.next_1d(), &pdf, &ur);
        a[ofs] += 1;
    }

    for (int i = 0; i < 10; ++i) {
        cout << a[i] << endl;
    }
}

void test_2d() {
    vector<float> weight;
    for(int i = 0; i < 25; ++i) {
        weight.push_back(i);
    }

    auto d2d = create_static_distrib2d<5,5>(weight.data());
//    auto d2d = create_static_alias_table2d<5,5>(weight.data());

    luminous::int2 offset;
    float p;
    auto uv = d2d.sample_continuous(luminous::make_float2(0.5, 0.5), &p, &offset);
    float pdf = d2d.PDF(uv);
    return;
}

int main() {




    test_2d();

    return 0;
}