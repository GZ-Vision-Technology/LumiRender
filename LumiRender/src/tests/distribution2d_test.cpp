//
// Created by Zero on 2021/7/28.
//

#include "render/distribution/distribution_handle.h"
#include "render/distribution/envmap_distribution.h"
using namespace luminous;
using namespace std;
int main() {
    vector<float> vec;
    int nu = 5;
    int nv = 5;
    for (int i = 0; i < nu * nv; ++i) {
        vec.push_back(i);
    }

    auto ed = EnvmapDistribution();
    ed.init(vec, nu, nv);
    ed.init_on_host();
    cout << "3245234" << endl;;
}