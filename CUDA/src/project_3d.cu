#include "types.h"
#include "project_3d.h"
#include "calc_weight.cuh"
#include "matrix_operator.cuh"

#include <iostream>
#include <cstdio>

float* Project3D(Point_3D* source,
    Point_3D* reciecer,
    float* projection,
    float* data,
    Geometry geo) {

    Point_3D* D_source;
    Point_3D* D_reciecer;
    float* D_data;

    gpuErrchk(cudaMalloc((void**)&D_source, geo.num_views * sizeof(Point_3D)));
    gpuErrchk(cudaMalloc((void**)&D_reciecer, geo.num_channels * sizeof(Point_3D)));
    gpuErrchk(cudaMalloc((void**)&D_data, geo.nx * geo.ny * geo.nz * sizeof(float)));

    gpuErrchk(cudaMemcpy(D_source, source, geo.num_views * sizeof(Point_3D), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_reciecer, reciecer, geo.num_channels * sizeof(Point_3D), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_data, data, geo.nx * geo.ny * geo.nz * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(D_geo, &geo, sizeof(struct Geometry)));
    gpuErrchk(cudaPeekAtLastError());
    
    float* weight = (float*)malloc(geo.nx * geo.ny * geo.nz * sizeof(float));
    float* D_weight;
    gpuErrchk(cudaMalloc((void**)&D_weight, geo.nx * geo.ny * geo.nz * sizeof(float)));

    dim3 block(32, 16, 1);
    dim3 grid((geo.nx + block.x - 1) / block.x, (geo.ny + block.y - 1) / block.y, (geo.nz + block.z - 1) / block.z);

    unsigned long index = 0;
    //float sum = 0;
    for (size_t i = 0; i < geo.num_views; i++)
    {
        for (size_t j = 0; j < geo.num_channels; j++) {

            memset(weight, 0, geo.nx * geo.ny * geo.nz * sizeof(float));
            gpuErrchk(cudaMemset(D_weight, 0, geo.nx * geo.ny * geo.nz * sizeof(float)));
            
            CalcWeight_kernel << <grid, block >> > (D_weight, source[i], reciecer[j]);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            Mul_Matrix_kernel << <grid, block >> > (D_weight, D_data, D_weight, geo.nx, geo.ny, geo.nz);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            gpuErrchk(cudaMemcpy(weight, D_weight, geo.nx * geo.ny * geo.nz * sizeof(float), cudaMemcpyDeviceToHost));

            index = i * geo.num_channels + j;
            projection[index] = 0;
            for (size_t k = 0; k < geo.nx * geo.ny * geo.nz; k++)
            {
                projection[index] += weight[k];
            }
        }
    }
    gpuErrchk(cudaFree(D_weight));
    free(weight);

    //int retVal;
    //FILE* fp;
    //fp = fopen("test.bin", "w");
    //retVal = fwrite(projection, num_views * num_channels * sizeof(float), 1, fp);
    //retVal = fclose(fp);

    gpuErrchk(cudaFree(D_source));
    gpuErrchk(cudaFree(D_reciecer));
    gpuErrchk(cudaFree(D_data));

    return projection;
}