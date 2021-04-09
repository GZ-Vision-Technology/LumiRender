
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "graphics/lstd/lstd.h"
#include "graphics/common.h"
#include <iostream>
#include <memory>
#include "render/samplers/independent.cpp"
#include "render/samplers/sampler.cpp"

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

using ::lstd::Variant;

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

class BaseP : public Variant<Sub1*, Sub2*> {
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
    BaseP bp(new Sub1());
    BaseP bp2(new Sub2());

    printf("%d bp2 ++--  %d s2\n", bp2.fun2(8), s2.fun2(8));
    printf("%d bp1 ++--  %d s2\n", bp.fun2(8), s1.fun2(8));


//    auto bb =  bp.get();
//    delete bp2.get();
//
//    BaseP bp2 = &s2;
//
//    cout << bp.fun1() << endl;
//    cout << bp.fun2(9) << endl;
//
//    cout << bp2.fun1() << endl;
//    cout << bp2.fun2(9) << endl;
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
//    testVariant();
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
}


__global__ void test_sampler(luminous::Sampler sh) {
//        auto s = LCGSampler(6);
//        printf("%f \n", s.next_1d());
//        printf("%f \n", sh.next_1d());
//    printf("adsfadsf %f\n", sh.next_1d());
//    printf("adsfadsf \n");
//    auto s = luminous::LCGSampler(6);
//    auto handle = luminous::SamplerHandle(s);
//    printf("%f \n", s.next_1d());

        printf("%f \n", sh.next_1d());
}

void testsampler(luminous::Sampler sh) {
    printf("%f \n", sh.next_1d());
}

int main()
{
//    testVariant();


    auto config = luminous::SamplerConfig();
    config.type = "LCGSampler";
    config.spp = 9;
//    auto sampler = SamplerHandle::create(config).get<LCGSampler>();
    auto sampler = luminous::Sampler::create(config);

    test_sampler<<<1u, 5u>>>(sampler);
//    testsampler(sampler);
    cudaDeviceSynchronize();
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//           c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
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
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
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
