//
// Created by Zero on 2021/7/28.
//

#include "render/distribution/distribution_handle.h"
#include "render/distribution/envmap_distribution.h"
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

    auto ed = EnvmapDistribution();
    ed.init(vec, nu, nv);
    ed.init_on_host();
//    ed.init_on_device(device);
//    ed.synchronize_to_gpu();
    auto distrib = ed.get_distribution();

    cout << distrib.PDF(luminous::make_float2(0.5,0.9));

}