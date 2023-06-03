#include "types.h"
#include "backproject_3d.h"
#include "backproject_3d.cuh"
#include "calc_weight.cuh"
#include "matrix_operator.cuh"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include "mex.h"

using namespace std;

void projection_interp(Geometry geo, float* projection, Point_3D* reciecer, float* proj_inter, Point_3D* reciecer_inter, int mul) {
    // 投影数据插值
    const int row_m = mul;
    const int row = (int)(geo.num_channels_x * row_m);
    const int col_m = mul;
    const int col = (int)(geo.num_channels_y * col_m);
    const int srcimg_size = geo.num_channels_x * geo.num_channels_y * sizeof(float);
    const int dstimg_size = row * col * sizeof(float);
    const float x_a = 1.0 / row_m;
    const float y_a = 1.0 / col_m;

    float* D_proj_inter;
    gpuErrchk(cudaMalloc((void**)&D_proj_inter, dstimg_size * geo.num_views * geo.num_panel));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();                             //声明数据类型
    cudaArray* cuArray_src;                                                                         //定义CUDA数组
    gpuErrchk(cudaMallocArray(&cuArray_src, &channelDesc, geo.num_channels_x, geo.num_channels_y));            //分配大小为col*row的CUDA数组
    tex_src.addressMode[0] = cudaAddressModeWrap;                       //寻址方式
    tex_src.addressMode[1] = cudaAddressModeWrap;                       //寻址方式 如果是三维数组则设置texRef.addressMode[2]
    tex_src.normalized = false;                                         //是否对纹理坐标归一化
    tex_src.filterMode = cudaFilterModeLinear;                          //硬件插值方式：最邻近插值--cudaFilterModePoint 双线性插值--cudaFilterModeLinear
    gpuErrchk(cudaBindTextureToArray(&tex_src, cuArray_src, &channelDesc));        //把CUDA数组绑定到纹理内存

    dim3 Block_resize(16, 16);
    dim3 Grid_resize((row + Block_resize.x - 1) / Block_resize.x, (col + Block_resize.y - 1) / Block_resize.y);

    for (size_t i = 0; i < geo.num_views * geo.num_panel; i++)
    {
        gpuErrchk(cudaMemcpyToArray(cuArray_src, 0, 0, projection + i * geo.num_channels, srcimg_size, cudaMemcpyHostToDevice));   //把源图像数据拷贝到CUDA数组

        // 调用核函数
        resize_proj_ker << <Grid_resize, Block_resize >> > (row, col, x_a, y_a, D_proj_inter + i * geo.num_channels * row_m * col_m);
        gpuErrchk(cudaPeekAtLastError());
    }
    gpuErrchk(cudaFreeArray(cuArray_src));
    gpuErrchk(cudaUnbindTexture(tex_src));

    // 探测器通道插值
    Point_3D* D_reciecer_inter;
    gpuErrchk(cudaMalloc((void**)&D_reciecer_inter, row * col * sizeof(Point_3D) * geo.num_panel));

    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();                   //声明数据类型
    cudaArray* cuArray_x;                                                                   //定义CUDA数组
    cudaArray* cuArray_y;
    cudaArray* cuArray_z;
    gpuErrchk(cudaMallocArray(&cuArray_x, &channelDesc, geo.num_channels_x, geo.num_channels_y));      //分配大小为col*row的CUDA数组
    gpuErrchk(cudaMallocArray(&cuArray_y, &channelDesc, geo.num_channels_x, geo.num_channels_y));
    gpuErrchk(cudaMallocArray(&cuArray_z, &channelDesc, geo.num_channels_x, geo.num_channels_y));
    tex_x.addressMode[0] = cudaAddressModeWrap;                                             //寻址方式
    tex_y.addressMode[0] = cudaAddressModeWrap;
    tex_z.addressMode[0] = cudaAddressModeWrap;
    tex_x.addressMode[1] = cudaAddressModeWrap;                                             //寻址方式 如果是三维数组则设置texRef.addressMode[2]
    tex_y.addressMode[1] = cudaAddressModeWrap;
    tex_z.addressMode[1] = cudaAddressModeWrap;
    tex_x.normalized = false;                                                               //是否对纹理坐标归一化
    tex_y.normalized = false;
    tex_z.normalized = false;
    tex_x.filterMode = cudaFilterModeLinear;                                                //硬件插值方式：最邻近插值--cudaFilterModePoint 双线性插值--cudaFilterModeLinear
    tex_y.filterMode = cudaFilterModeLinear;
    tex_z.filterMode = cudaFilterModeLinear;
    gpuErrchk(cudaBindTextureToArray(&tex_x, cuArray_x, &channelDesc));                                //把CUDA数组绑定到纹理内存
    gpuErrchk(cudaBindTextureToArray(&tex_y, cuArray_y, &channelDesc));
    gpuErrchk(cudaBindTextureToArray(&tex_z, cuArray_z, &channelDesc));

    float* reciecer_x = (float*)malloc(geo.num_channels * geo.num_panel * sizeof(float));
    float* reciecer_y = (float*)malloc(geo.num_channels * geo.num_panel * sizeof(float));
    float* reciecer_z = (float*)malloc(geo.num_channels * geo.num_panel * sizeof(float));
    for (size_t i = 0; i < geo.num_channels * geo.num_panel; i++)
    {
        reciecer_x[i] = reciecer[i].x;
        reciecer_y[i] = reciecer[i].y;
        reciecer_z[i] = reciecer[i].z;
    }
    for (size_t i = 0; i < geo.num_panel; i++)
    {
        gpuErrchk(cudaMemcpyToArray(cuArray_x, 0, 0, reciecer_x + i * geo.num_channels, geo.num_channels * sizeof(float), cudaMemcpyHostToDevice));   //把数据拷贝到CUDA数组
        gpuErrchk(cudaMemcpyToArray(cuArray_y, 0, 0, reciecer_y + i * geo.num_channels, geo.num_channels * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpyToArray(cuArray_z, 0, 0, reciecer_z + i * geo.num_channels, geo.num_channels * sizeof(float), cudaMemcpyHostToDevice));

        resize_reciecer_ker << <Grid_resize, Block_resize >> > (row, col, x_a, y_a, D_reciecer_inter + i * row * col);
        gpuErrchk(cudaPeekAtLastError());
    }
    gpuErrchk(cudaFreeArray(cuArray_x));
    gpuErrchk(cudaFreeArray(cuArray_y));
    gpuErrchk(cudaFreeArray(cuArray_z));
    gpuErrchk(cudaUnbindTexture(tex_x));
    gpuErrchk(cudaUnbindTexture(tex_y));
    gpuErrchk(cudaUnbindTexture(tex_z));
    free(reciecer_x);
    free(reciecer_y);
    free(reciecer_z);

    gpuErrchk(cudaMemcpy(reciecer_inter, D_reciecer_inter, geo.num_channels * row_m * col_m * geo.num_panel * sizeof(Point_3D), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(proj_inter, D_proj_inter, geo.num_views * geo.num_channels * row_m * col_m * geo.num_panel * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(D_reciecer_inter));
    gpuErrchk(cudaFree(D_proj_inter));

}

float* Backproject3D(Point_3D* source,
    Point_3D* reciecer,
    float* projection,
    float* data,
    Geometry geo) {

    Point_3D* D_source;
    Point_3D* D_reciecer;
    float* D_data;

    gpuErrchk(cudaMalloc((void**)&D_source, geo.num_views * geo.num_panel * sizeof(Point_3D)));
    gpuErrchk(cudaMalloc((void**)&D_reciecer, geo.num_channels * geo.num_panel * sizeof(Point_3D)));
    gpuErrchk(cudaMalloc((void**)&D_data, geo.nx * geo.ny * geo.nz * sizeof(float)));

    gpuErrchk(cudaMemcpy(D_source, source, geo.num_views * geo.num_panel * sizeof(Point_3D), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_reciecer, reciecer, geo.num_channels * geo.num_panel * sizeof(Point_3D), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(D_geo, &geo, sizeof(struct Geometry)));
    gpuErrchk(cudaPeekAtLastError());

    int n = geo.interp;
    float* proj_inter = (float*)malloc(geo.num_views * geo.num_channels * n * n * geo.num_panel * sizeof(float));
    Point_3D* reciecer_inter = (Point_3D*)malloc(geo.num_channels * n * n * geo.num_panel * sizeof(Point_3D));
    projection_interp(geo, projection, reciecer, proj_inter, reciecer_inter, n);




    //ofstream ofs;
    //ofs.open("Point_inter.bin", ios::out | ios::binary);
    //ofs.write((const char*)test, geo.num_panel * dstimg_size);
    //ofs.close();

    //float* test = (float*)malloc(dstimg_size * geo.num_views * geo.num_panel);
    //gpuErrchk(cudaMemcpy(test, D_proj_inter, dstimg_size * geo.num_views * geo.num_panel, cudaMemcpyDeviceToHost));
    //ofs.open("proj_interp.bin", ios::out | ios::binary);
    //ofs.write((const char*)test, dstimg_size * geo.num_views * geo.num_panel);
    //ofs.close();

    float* weight = (float*)malloc(geo.nx * geo.ny * geo.nz * sizeof(float));
    float* D_weight;
    float* D_weight_proj;
    float* D_weight_sum;
    gpuErrchk(cudaMalloc((void**)&D_weight, geo.nx * geo.ny * geo.nz * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&D_weight_proj, geo.nx * geo.ny * geo.nz * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&D_weight_sum, geo.nx * geo.ny * geo.nz * sizeof(float)));
    gpuErrchk(cudaMemset(D_weight_sum, 0, geo.nx * geo.ny * geo.nz * sizeof(float)));

    dim3 block(32, 16, 1);
    dim3 grid((geo.nx + block.x - 1) / block.x, (geo.ny + block.y - 1) / block.y, (geo.nz + block.z - 1) / block.z);

    unsigned long index = 0;
    float sum = 0;
    for (size_t k = 0; k < geo.num_panel; k++)
    {
        for (size_t i = 0; i < geo.num_views; i++)
        {
            for (size_t j = 0; j < geo.num_channels * n * n; j++) {

                //mexPrintf("(%d, %d, %d)\n", i, j, k);
                index = k * geo.num_views * geo.num_channels * n * n + i * geo.num_channels * n * n + j;

                memset(weight, 0, geo.nx * geo.ny * geo.nz * sizeof(float));
                gpuErrchk(cudaMemset(D_weight, 0, geo.nx * geo.ny * geo.nz * sizeof(float)));

                CalcWeight_kernel << <grid, block >> > (D_weight, source[i + k * geo.num_views], reciecer_inter[j + k * geo.num_channels * n * n]);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());

                Add_Matrix_kernel << <grid, block >> > (D_weight_sum, D_weight, D_weight_sum, geo.nx, geo.ny, geo.nz);
                gpuErrchk(cudaPeekAtLastError());

                gpuErrchk(cudaMemcpy(D_weight_proj, D_weight, geo.nx * geo.ny * geo.nz * sizeof(float), cudaMemcpyDeviceToDevice));
                MatrixSum << <ceil((double)(geo.nx * geo.ny * geo.nz) / THREAD_LENGTH), THREAD_LENGTH >> > (D_weight, geo.nx * geo.ny * geo.nz);
                //gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaMemcpy(weight, D_weight, geo.nx * geo.ny * geo.nz * sizeof(float), cudaMemcpyDeviceToHost));
                sum = 0;
                for (int l = 0; l < ceil((double)(geo.nx * geo.ny * geo.nz) / THREAD_LENGTH); ++l) {
                    sum += weight[l * THREAD_LENGTH];    //对每个块内部分和求和
                }

                Mul_Matrix_kernel << <grid, block >> > (D_weight_proj, D_weight_proj, proj_inter[index], geo.nx, geo.ny, geo.nz);
                gpuErrchk(cudaPeekAtLastError());

                Data_Sum_kernel << <grid, block >> > (D_data, D_weight_proj, sum, geo.nx, geo.ny, geo.nz);
                gpuErrchk(cudaPeekAtLastError());
                
                //gpuErrchk(cudaMemcpy(data, D_data, geo.nx * geo.ny * geo.nz * sizeof(float), cudaMemcpyDeviceToHost))
                //ofstream ofs;
                //ofs.open("test_interp.bin", ios::out | ios::binary);
                //ofs.write((const char*)data, geo.nx* geo.ny* geo.nz * sizeof(float));
                //ofs.close();
            }
        }
    }
    //mexPrintf("AA\n");
    
    //gpuErrchk(cudaMemcpy(data, D_weight_sum, geo.nx * geo.ny * geo.nz * sizeof(float), cudaMemcpyDeviceToHost))

    //ofstream ofs;
    //ofs.open("test_interp.bin", ios::out | ios::binary);
    //ofs.write((const char*)data, geo.nx * geo.ny * geo.nz * sizeof(float));
    //ofs.close();

    // 反投影数据除以总走时
    Div_Matrix_kernel << <grid, block >> > (D_data, D_weight_sum, D_data, geo.nx, geo.ny, geo.nz);
    gpuErrchk(cudaPeekAtLastError());
    
    gpuErrchk(cudaFree(D_weight_proj));
    gpuErrchk(cudaFree(D_source));
    gpuErrchk(cudaFree(D_reciecer));
    free(weight);
    //mexPrintf("BB\n");
    
    gpuErrchk(cudaMemcpy(data, D_data, geo.nx * geo.ny* geo.nz * sizeof(float), cudaMemcpyDeviceToHost))

    //ofstream ofs;
    //ofs.open("test_interp.bin", ios::out | ios::binary);
    //ofs.write((const char*)data, geo.nx* geo.ny* geo.nz * sizeof(float));
    //ofs.close();
    //mexPrintf("BB\n");

    gpuErrchk(cudaFree(D_weight));
    gpuErrchk(cudaFree(D_data));

    return data;
}