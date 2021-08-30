//
// Created by Zero on 2021/7/28.
//

#include "render/include/distribution_mgr.h"
#include "gpu/framework/cuda_impl.h"
#include "core/backend/managed.h"


using namespace luminous;
using namespace std;
int main() {

    auto device = create_cuda_device();

    vector<float> vec;
    int nu = 5;
    int nv = 5;
    for (int i = 0; i < nu * nv; ++i) {
        vec.push_back(i);
    }



}