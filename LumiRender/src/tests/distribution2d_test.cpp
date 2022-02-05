//
// Created by Zero on 2021/7/28.
//

#include "render/include/distribution_mgr.h"
#include "gpu/framework/cuda_impl.h"
#include "core/backend/managed.h"
#include "base_libs/sampling/alias_table.h"
#include "base_libs/sampling/distribution.h"

using namespace luminous;
using namespace std;
int main() {

    auto[table, PDF] = luminous::create_alias_table(vector<float>({1,1,1}));

    AliasData alias_data{BufferView(table.data(), table.size()), BufferView(PDF.data(), PDF.size())};

    TAliasTable alias_table(alias_data);

    float ur, pdf;
    int ofs;
    float u = 0.1;
    auto uc = alias_table.sample_discrete(u, &pdf, &ur);

    auto builder = Distribution1D::create_builder({1,2,3});

    Distribution1D distribution(BufferView<const float>(builder.func.data(), builder.func.size()),
                             BufferView<const float>(builder.CDF.data(), builder.CDF.size()), builder.func_integral);

    float ur2, pdf2;
    int ofs2;
    auto uc2 = distribution.sample_discrete(u, &pdf2, &ur2);

    return 0;
}