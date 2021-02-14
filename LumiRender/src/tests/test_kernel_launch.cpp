//
// Created by Zero on 2020/12/29.
//

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <optix.h>
#include <string>
#include <driver_types.h>

using namespace std;

extern "C" char ptxCode[];

int main() {


    string s = ptxCode;

    CUmodule module;
    cuModuleLoadData(&module, ptxCode);
    CUfunction func;
    cuModuleGetFunction(&func, module, "addKernel");

    cout << s;
    return 0;
}