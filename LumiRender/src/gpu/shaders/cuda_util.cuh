//
// Created by Zero on 20/10/2021.
//


#pragma once


/**
 * num of grid is 1, num of block is 1
 * @return
 */
__device__ int task_id_g1_b1() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    return threadId;
}

/**
 * num of grid is 1, num of block is 2
 * @return
 */
__device__ int task_id_g1_b2() {
    int threadId = blockIdx.x * blockDim.x * blockDim.y
                   + threadIdx.y * blockDim.x + threadIdx.x;
    return threadId;
}

__device__ int task_id_g1_b3() {
    int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
            + threadIdx.z * blockDim.y * blockDim.x
            + threadIdx.y * blockDim.x + threadIdx.x;
    return threadId;
}

__device__ int task_id_g2_b1() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

__device__ int task_id_g2_b2() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ int task_id_g2_b3() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ int task_id_g3_b1() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

__device__ int task_id_g3_b2() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ int task_id_g3_b3() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}