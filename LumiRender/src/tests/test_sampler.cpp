//
// Created by Zero on 2021/2/28.
//

#include "iostream"
#include "render/include/sampler.h"
//#include "render/include/scene_graph.h"

using namespace luminous;
using namespace std;

int main() {
    auto config = SamplerConfig();
    config.type = "PCGSampler";
    config.spp = 9;
    auto sampler = SamplerHandler::create(config);
    cout << sampler.to_string() << endl;

}