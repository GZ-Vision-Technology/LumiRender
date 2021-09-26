//
// Created by Zero on 2021/2/14.
//


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "base_libs/lstd/lstd.h"
#include "base_libs/common.h"
#include <stdio.h>
#include <iostream>
#include "render/samplers/sampler.cpp"
#include "render/samplers/independent.cpp"
#include <cuda.h>
#include <cuda/atomic>
#include "render/lights/shader_include.h"
#include "tests/test_light.h"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

class Sub1 {
public:
    XPU int fun1() {
        return 0;
    }

    XPU int fun2(int a) {
        return a;
    }
};

class Sub2 {
public:
    XPU int fun1() {
        return 1;
    }

    XPU int fun2(int a) {
        return 2 * a;
    }
};

using luminous::lstd::Variant;

class Base : public Variant<Sub1, Sub2> {
public:
    using Variant::Variant;

    XPU int fun1() {
        return dispatch([](auto &&arg) { return arg.fun1(); });
    }

    XPU int fun2(int a) {
        LUMINOUS_VAR_DISPATCH(fun2, a);
    }
};

class BaseP : public Variant<Sub1 *, Sub2 *> {
public:
    using Variant::Variant;

    XPU int fun1() {
        return dispatch([](auto &&arg) { return arg->fun1(); });
    }

    XPU int fun2(int a) {
        LUMINOUS_VAR_PTR_DISPATCH(fun2, a);
    }
};



XPU void testVariant() {
    using namespace std;

    Sub1 s1 = Sub1();
    Sub2 s2 = Sub2();

//    printf("%d s--\n", s1.fun1());
//    printf("%d s2--\n", s2.fun1());

    Base b(s1);

    Base b2(s2);
    printf("%d b1--  %d s1\n", b.fun1(), s1.fun1());
    printf("%d b2--  %d s2\n", b2.fun1(), s2.fun1());
    printf("%d b1 ++--  %d s1\n", b.fun2(9), s1.fun2(9));
    printf("%d b2 ++--  %d s2\n", b2.fun2(8), s2.fun2(8));

//
//    cout << sizeof(b) << endl;
//    cout << sizeof(s2) << endl;
//
////
//    cout << b.fun1() << endl;
//    cout << b.fun2(9) << endl;
//
//    BaseP bp = &s1;
//
//    BaseP bp2 = &s2;
//
//    cout << bp.fun1() << endl;
//    cout << bp.fun2(9) << endl;
//
//    cout << bp2.fun1() << endl;
//    cout << bp2.fun2(9) << endl;
}


extern "C" {
    __global__ void addKernel(int *c, const int *a, const int *b) {
        int i = threadIdx.x;
        c[i] = a[i] + b[i];
//        testVariant();
        printf("C:%d, B:%d, A: %d\n", c[i], b[i], a[i]);
    }

    __global__ void testKernel(int *c) {
        printf("%d \n", threadIdx.x);
    }

    __global__ void test_tex_sample(CUtexObject handle, float u, float v) {
//        auto val = tex2D<uint8_t>(handle, 0, v);
//        auto val2 = tex2D<uint8_t>(handle, 1, v);
        auto val = tex2D<float>(handle, 0, v);
        auto val2 = tex2D<float>(handle, 1, v);
//        printf("%d,%d,%d,%d\n", (uint32_t)val.x,(uint32_t)val.y,(uint32_t)val.z,(uint32_t)val.w);
//        printf("tex2D[0] :%u, tex2D[1] : %u\n",(uint32_t)val,(uint32_t)val2);
        printf("tex2D[0] :%f, tex2D[1] : %f\n",val,val2);
    }


    __global__ void test_light(luminous::Light*light) {
        using namespace luminous;
//        light.print();
        printf("%f\n", light->get<AreaLight>()->padded);
    }

    __global__ void test_area_light(luminous::AreaLight*light) {
        using namespace luminous;
        //        light.print();
        printf("%f\n", light->padded);
    }

    __global__ void test_AL(luminous::AL * light) {
        using namespace luminous;
        //        light.print();
        printf("%f    %d\n", light->padded, light->_type);
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **) &dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
