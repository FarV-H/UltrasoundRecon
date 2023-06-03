#pragma once

#include "cuda_utils.h"

#define THREAD_LENGTH 256

// C = A + B
__global__ void Add_Matrix_kernel(float* A, float* B, float* C, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    C[index] = A[index] + B[index];
}

// C = A - B
__global__ void Sub_Matrix_kernel(float* A, float* B, float* C, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    C[index] = A[index] - B[index];
}

// C = A * B
__global__ void Mul_Matrix_kernel(float* A, float* B, float* C, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    C[index] = A[index] * B[index];
}
// A = w * B
__global__ void Mul_Matrix_kernel(float* A, float* B, float w, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    A[index] = w * B[index];
}

// C = A / B
__global__ void Div_Matrix_kernel(float* A, float* B, float* C, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    if (B[index] == 0)
    {
        C[index] = 0;
        return;
    }

    C[index] = A[index] / B[index];
}

// A = w / B
__global__ void Div_Matrix_kernel(float* A, float* B, float w, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    if (B[index] == 0)
    {
        A[index] = 0;
        return;
    }

    A[index] = w / B[index];
}

// 矩阵求和
__global__ void MatrixSum(float* d_A, int n) {
    unsigned int t = threadIdx.x;
    __shared__ float partialSum[THREAD_LENGTH];
    if (blockIdx.x * blockDim.x + t < n)
        partialSum[t] = d_A[blockIdx.x * blockDim.x + t];
    else
        partialSum[t] = 0;
    __syncthreads();  //将数组加载到共享存储器。
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (t % (2 * stride) == 0)
            partialSum[t] += partialSum[t + stride];
        __syncthreads();
    }
    /*将每个线程块内 threadIdx.x 为零的线程中的值传回d_A。
    主机函数中将会对这几个线程求和，以得到最终的和。*/
    if (t == 0)
        d_A[blockIdx.x * blockDim.x + t] = partialSum[t];
}