//
// Created by Zero on 2021/7/28.
//

#include "render/include/distribution_mgr.h"
#include "gpu/framework/cuda_impl.h"
#include "core/backend/managed.h"
#include "base_libs/sampling/alias_table.h"

using namespace luminous;
using namespace std;
int main() {

    float weight[3] = {1,5,2};

    auto[table, PDF] = luminous::create_alias_table(BufferView<float>(weight));

    return 0;
}