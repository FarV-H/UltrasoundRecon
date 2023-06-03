#pragma once


#define INTERP_WINDOW 2

//__device__ float* D_distance_map;
//__device__ Point_3D* D_point_map;
// 
//定义纹理内存变量
texture<float, cudaTextureType2D, cudaReadModeElementType>  tex_src;
texture<float, cudaTextureType2D, cudaReadModeElementType>  tex_x;
texture<float, cudaTextureType2D, cudaReadModeElementType>  tex_y;
texture<float, cudaTextureType2D, cudaReadModeElementType>  tex_z;


__global__ void resize_proj_ker(int row, int col, float x_a, float y_a, float* out)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;  //col
    int y = threadIdx.y + blockDim.y * blockIdx.y;  //row

    if (x < row && y < col)
    {
        float xx = x * x_a;
        float yy = y * y_a;
        //这里的xx和yy都是浮点数，tex2D函数返回的数值已经过硬件插值了，所以不需要开发者再进行插值啦~
        out[y * row + x] = tex2D(tex_src, xx, yy);
    }
}

__global__ void resize_reciecer_ker(int row, int col, float x_a, float y_a, Point_3D* out)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;  //col
    int y = threadIdx.y + blockDim.y * blockIdx.y;  //row

    if (x < row && y < col)
    {
        float xx = x * x_a;
        float yy = y * y_a;
        //这里的xx和yy都是浮点数，tex2D函数返回的数值已经过硬件插值了，所以不需要开发者再进行插值啦~
        out[y * row + x].x = tex2D(tex_x, xx, yy);
        out[y * row + x].y = tex2D(tex_y, xx, yy);
        out[y * row + x].z = tex2D(tex_z, xx, yy);
    }
}

__global__ void Distance_Map_kernel(float* D_distance_map, Point_3D* D_point_map, unsigned long nx, unsigned long ny, unsigned long nz) {
    int Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;
    //printf("(%d, %d, %d, %d)\n", Xidx, Yidx, Zidx, index);

    if ((Xidx == INTERP_WINDOW) && (Yidx == INTERP_WINDOW) && (Zidx == INTERP_WINDOW)) {
        D_distance_map[index] = 0;
        D_point_map[index] = Point_3D((float)(Xidx - INTERP_WINDOW), (float)(Yidx - INTERP_WINDOW), (float)(Zidx - INTERP_WINDOW));
        return;
    }

    float weight = sqrt((float)((Zidx - INTERP_WINDOW) * (Zidx - INTERP_WINDOW)
                                + (Yidx - INTERP_WINDOW) * (Yidx - INTERP_WINDOW)
                                + (Xidx - INTERP_WINDOW) * (Xidx - INTERP_WINDOW)));
    D_distance_map[index] = 1 / weight;
    D_point_map[index] = Point_3D((float)(Xidx - INTERP_WINDOW), (float)(Yidx - INTERP_WINDOW), (float)(Zidx - INTERP_WINDOW));
}

__global__ void Distance_Sort_kernel(float* D_distance_pre, Point_3D* D_point_pre, float* D_distance_map, Point_3D* D_point_map, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    unsigned long sum = 0;
    for (size_t i = 0; i < nx * ny * nz; i++)
    {
        if (D_distance_pre[i] > D_distance_pre[index])
        {
            sum++;
        }
        if (D_distance_pre[i] == D_distance_pre[index] && i < index)
        {
            sum++;
        }
    }
    D_distance_map[sum] = D_distance_pre[index];
    D_point_map[sum] = D_point_pre[index];
}

__global__ void Data_Sum_kernel(float* D_data, float* D_weight_proj, float weight, unsigned long nx, unsigned long ny, unsigned long nz) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
        return;
    }

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    if (weight != 0)
    {
        D_data[index] += D_weight_proj[index] / weight;
    }
    else {
        D_data[index] += 0;
    }

}

//__global__ void interp_proj_kernel(Point_3D *D_reciecer, float *D_proj, Point_3D* D_reciecer_interp, float *D_proj_interp, unsigned long nx, unsigned long ny, unsigned long nz, unsigned long unsample) {
//    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
//    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;
//
//    unsigned long index = Zidx * nx * unsample * ny * unsample + Yidx * nx * unsample + Xidx;
//
//    if (Xidx >= nx * unsample || Yidx >= ny * unsample || Zidx >= nz)
//    {
//        return;
//    }
//
//    float u = (Xidx % unsample) / unsample;
//    float v = (Yidx % unsample) / unsample;
//    D_proj_interp[index] = (1 - u) * (1 - v) * D_proj[Zidx * nx * ny + (Yidx / unsample) * nx + (Xidx / unsample)]
//                            + u * (1 - v) * D_proj[Zidx * nx * ny + (Yidx / unsample) * nx + ((Xidx / unsample) + 1)]
//                            + (1 - u) * v * D_proj[Zidx * nx * ny + ((Yidx / unsample) + 1) * nx + (Xidx / unsample)]
//                            + u * v * D_proj[Zidx * nx * ny + ((Yidx / unsample) + 1) * nx + ((Xidx / unsample) + 1)];
//    D_reciecer_interp[index].x = (1 - u) * (1 - v) * D_proj[Zidx * nx * ny + (Yidx / unsample) * nx + (Xidx / unsample)]
//        + u * (1 - v) * D_proj[Zidx * nx * ny + (Yidx / unsample) * nx + ((Xidx / unsample) + 1)]
//        + (1 - u) * v * D_proj[Zidx * nx * ny + ((Yidx / unsample) + 1) * nx + (Xidx / unsample)]
//        + u * v * D_proj[Zidx * nx * ny + ((Yidx / unsample) + 1) * nx + ((Xidx / unsample) + 1)];
//}

__global__ void Interp_kernel(float* D_data, float* D_interpData, float* D_mask, float* D_distance_map, Point_3D* D_point_map, unsigned long nx, unsigned long ny, unsigned long nz, bool *flag) {
    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned long index = Zidx * nx * ny + Yidx * nx + Xidx;

    //if (Xidx >= nx || Yidx >= ny || Zidx >= nz) {
    //    return;
    //}
    //if (D_mask[index] != 0)
    //{
    //    D_interpData[index] = D_data[index];
    //    return;
    //}
    //else {
    //    D_interpData[index] = 0;
    //}

    //float x_1 = 0;
    //float x_2 = 0;
    //float y_1 = 0;
    //float y_2 = 0;
    //float z_1 = 0;
    //float z_2 = 0;
    //float weight_x_1 = 0;
    //float weight_x_2 = 0;
    //float weight_y_1 = 0;
    //float weight_y_2 = 0;
    //float weight_z_1 = 0;
    //float weight_z_2 = 0;
    //bool flag = true;
    //for (size_t i = 0; flag; i++)
    //{
    //    flag = false;
    //    if ((weight_x_1 == 0) && (Xidx + i) < nx)
    //    {
    //        if (D_mask[Zidx * nx * ny + Yidx * nx + Xidx + i] != 0)
    //        {
    //            weight_x_1 = i;
    //            x_1 = D_data[Zidx * nx * ny + Yidx * nx + Xidx + i];
    //        }
    //    }
    //    if ((weight_x_2 == 0) && (Xidx - i) >= 0)
    //    {
    //        if (D_mask[Zidx * nx * ny + Yidx * nx + Xidx - i] != 0)
    //        {
    //            weight_x_2 = i;
    //            x_2 = D_data[Zidx * nx * ny + Yidx * nx + Xidx - i];
    //        }
    //    }
    //    if ((weight_y_1 == 0) && (Yidx + i) < ny)
    //    {
    //        if (D_mask[Zidx * nx * ny + (Yidx + i) * nx + Xidx] != 0)
    //        {
    //            weight_y_1 = i;
    //            y_1 = D_data[Zidx * nx * ny + (Yidx + i) * nx + Xidx];
    //        }
    //    }
    //    if ((weight_y_2 == 0) && (Yidx - i) >= 0)
    //    {
    //        if (D_mask[Zidx * nx * ny + (Yidx - i) * nx + Xidx] != 0)
    //        {
    //            weight_y_2 = i;
    //            y_2 = D_data[Zidx * nx * ny + (Yidx - i) * nx + Xidx];
    //        }
    //    }
    //    if ((weight_z_1 == 0) && (Zidx + i) < nz)
    //    {
    //        if (D_mask[(Zidx + i) * nx * ny + Yidx * nx + Xidx] != 0)
    //        {
    //            weight_z_1 = i;
    //            z_1 = D_data[(Zidx + i) * nx * ny + Yidx * nx + Xidx];
    //        }
    //    }
    //    if ((weight_z_2 == 0) && (Zidx - i) >= 0)
    //    {
    //        if (D_mask[(Zidx - i) * nx * ny + Yidx * nx + Xidx] != 0)
    //        {
    //            weight_z_2 = i;
    //            z_2 = D_data[(Zidx - i) * nx * ny + Yidx * nx + Xidx];
    //        }
    //    }

    //    if (weight_x_1 * weight_x_2 * weight_y_1 * weight_y_2 * weight_z_1 * weight_z_2 == 0)
    //    {
    //        flag = true;
    //    }
    //    if (((Xidx + i) < nx) && ((Xidx - i) >= 0) &&
    //        ((Yidx + i) < ny) && ((Yidx - i) >= 0) &&
    //        ((Zidx + i) < nz) && ((Zidx - i) >= 0))
    //    {
    //        flag = false;
    //    }
    //}






    const int num = (INTERP_WINDOW * 2 + 1) * (INTERP_WINDOW * 2 + 1) * (INTERP_WINDOW * 2 + 1);
    float weight[num];
    float val[num];
    float weight_temp = 0;

    for (size_t i = 0; i < num; i++)
    {
        if ((Zidx + D_point_map[i].z) < nz && (Zidx + D_point_map[i].z) >= 0 &&
            (Yidx + D_point_map[i].y) < ny && (Yidx + D_point_map[i].y) >= 0 &&
            (Xidx + D_point_map[i].x) < nx && (Xidx + D_point_map[i].x) >= 0) {
            if (D_mask[(Zidx + int(D_point_map[i].z)) * nx * ny + (Yidx + int(D_point_map[i].y)) * nx + (Xidx + int(D_point_map[i].x))] != 0)
            {
                weight[i] = D_distance_map[i];
                val[i] = D_data[(Zidx + int(D_point_map[i].z)) * nx * ny + (Yidx + int(D_point_map[i].y)) * nx + (Xidx + int(D_point_map[i].x))];
            }
            else
            {
                weight[i] = 0;
                val[i] = D_data[(Zidx + int(D_point_map[i].z)) * nx * ny + (Yidx + int(D_point_map[i].y)) * nx + (Xidx + int(D_point_map[i].x))];
            }
        }
        else {
            weight[i] = 0;
            val[i] = 0;// D_data[(Zidx + int(D_point_map[i].z)) * nx * ny + (Yidx + int(D_point_map[i].y)) * nx + (Xidx + int(D_point_map[i].x))];
        }
        if (weight[i] > weight_temp)
        {
            weight_temp = weight[i];
        }
    }
    if (weight_temp < 1)
    {
        flag[0] = true;
        return;
    }

    D_interpData[index] = 0;
    weight_temp = 0;
    for (int i = 0; i < num; i++)
    {
        D_interpData[index] += (val[i] * weight[i]);
        weight_temp += weight[i];
    }
    //printf("(%d, %f)\n", num, weight_temp);

    //if (Xidx == 100)
    //{
    //    printf("(%d, %d, %d) %f, %f\n", Xidx, Yidx, Zidx, D_interpData[index], weight_temp);
    //}

    if (weight_temp != 0)
    {
        D_interpData[index] = D_interpData[index] / weight_temp;
    }
    else {
        D_interpData[index] = 0;
    }
}
