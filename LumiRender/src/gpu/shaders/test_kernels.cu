//
// Created by Zero on 2021/2/14.
//



#include "cuda_util.cuh"

extern "C" __global__ void addKernel(int _, int __,int *a, int *b, int *c)
{
    int i = threadIdx.x;
    if (i >= 5) {
        return;
    }
    c[i] = a[i] + b[i];
    printf("%d\n", a[i]);
}