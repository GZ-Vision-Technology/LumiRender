//
// Created by Zero on 2020/12/29.
//

#include <iostream>
//#include "cuda.h"
#include "gpu/cuda/jitify/jitify.hpp"
//#include "core/math/math_util.h"
#include "core/math/ray_hit.h"

using namespace std;

//void test() {
//    const char* program1 =
//            "my_program1\n"
//            "#include \"example_headers/my_header1.cuh\"\n"
//            "#include \"example_headers/my_header2.cuh\"\n"
//            "#include \"example_headers/my_header3.cuh\"\n"
//            "#include \"example_headers/my_header4.cuh\"\n"
//            "\n"
//            "__global__\n"
//            "void my_kernel1(float const* indata, float* outdata) {\n"
//            "    outdata[0] = indata[0] + 1;\n"
//            "    outdata[0] -= 1;\n"
//            "}\n"
//            "\n"
//            "template<int C, typename T>\n"
//            "__global__\n"
//            "void my_kernel2(float const* indata, float* outdata) {\n"
//            "    for( int i=0; i<C; ++i ) {\n"
//            "        outdata[0] = "
//            "pointless_func(identity(sqrt(square(negate(indata[0])))));\n"
//            "    }\n"
//            "}\n";
//
//    using jitify::reflection::instance_of;
//    using jitify::reflection::NonType;
//    using jitify::reflection::reflect;
//    using jitify::reflection::Type;
//    using jitify::reflection::type_of;
//
//    thread_local static jitify::JitCache kernel_cache;
//}

using namespace std;

int main() {
    int *p;
//    cudaMalloc((void**)&p, 9);
}