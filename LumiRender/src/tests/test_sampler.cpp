//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/include/sampler.h"
#include "tuple"
//#include "render/include/scene_graph.h"

using namespace luminous;
using namespace std;

int main() {

    using types = tuple<int , string>;
    auto config = SamplerConfig();
    config.type = "LCGSampler";
    config.spp = 9;
    auto sampler = SamplerHandle::create(config);

    cout << sampler.to_string() << endl;

}