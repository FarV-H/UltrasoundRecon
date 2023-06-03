#pragma once

#include "types.h"
#include "cuda_utils.h"

__constant__ struct Geometry D_geo;

__global__ void CalcWeight_kernel(float* D_weight, Point_3D source, Point_3D reciecer) {

    unsigned long Xidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Yidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long Zidx = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned long index = Zidx * D_geo.nx * D_geo.ny + Yidx * D_geo.nx + Xidx;
    
    if (Xidx >= D_geo.nx || Yidx >= D_geo.ny || Zidx >= D_geo.nz)
    {
        return;
    }

    // 扫描收集射线经过的像素
    // 设定约束：Point(i,j,k)至射线距离小于半个像素）且不超出端点范围
    Point_3D p = Point_3D(Xidx * D_geo.dx, Yidx * D_geo.dy, Zidx * D_geo.dz);
    float length = (reciecer - source).norm();
    float distance = cross(reciecer - source, p - source).norm() / length;
    //if (Xidx == 12 && Zidx == 12) {
    //    printf("(%f, %f, %f, %f, %f, %f)\n", p.x, p.y, p.z, length, distance, (Point_3D(D_geo.dx, D_geo.dy, D_geo.dz).norm() / 2));
    //}
    
    Point_3D temp_points    = Point_3D(-1, -1, -1);
    Point_3D cross_points_1 = Point_3D(-1, -1, -1);
    Point_3D cross_points_2 = Point_3D(-1, -1, -1);

    if ((distance < (Point_3D(D_geo.dx, D_geo.dy, D_geo.dz).norm() / 2))
        && ((p - source).norm() <= length)
        && ((p - reciecer).norm() <= length)) {

        // 求射线与所经过路径的体像素交叉点
        temp_points.x = p.x + 0.5 * D_geo.dx;
        temp_points.y = (source.x * reciecer.y - source.y * reciecer.x + (p.x + 0.5 * D_geo.dx) * source.y - (p.x + 0.5 * D_geo.dx) * reciecer.y) / (source.x - reciecer.x);
        temp_points.z = (source.x * reciecer.z - source.z * reciecer.x + (p.x + 0.5 * D_geo.dx) * source.z - (p.x + 0.5 * D_geo.dx) * reciecer.z) / (source.x - reciecer.x);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = p.x - 0.5 * D_geo.dx;
        temp_points.y = (source.x * reciecer.y - source.y * reciecer.x + (p.x - 0.5 * D_geo.dx) * source.y - (p.x - 0.5 * D_geo.dx) * reciecer.y) / (source.x - reciecer.x);
        temp_points.z = (source.x * reciecer.z - source.z * reciecer.x + (p.x - 0.5 * D_geo.dx) * source.z - (p.x - 0.5 * D_geo.dx) * reciecer.z) / (source.x - reciecer.x);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.y * reciecer.x - source.x * reciecer.y + (p.y + 0.5 * D_geo.dy) * source.x - (p.y + 0.5 * D_geo.dy) * reciecer.x) / (source.y - reciecer.y);
        temp_points.y = p.y + 0.5 * D_geo.dy;
        temp_points.z = (source.y * reciecer.z - source.z * reciecer.y + (p.y + 0.5 * D_geo.dy) * source.z - (p.y + 0.5 * D_geo.dy) * reciecer.z) / (source.y - reciecer.y);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.y * reciecer.x - source.x * reciecer.y + (p.y - 0.5 * D_geo.dy) * source.x - (p.y - 0.5 * D_geo.dy) * reciecer.x) / (source.y - reciecer.y);
        temp_points.y = p.y - 0.5 * D_geo.dy;
        temp_points.z = (source.y * reciecer.z - source.z * reciecer.y + (p.y - 0.5 * D_geo.dy) * source.z - (p.y - 0.5 * D_geo.dy) * reciecer.z) / (source.y - reciecer.y);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.z * reciecer.x - source.x * reciecer.z + (p.z + 0.5 * D_geo.dz) * source.x - (p.z + 0.5 * D_geo.dz) * reciecer.x) / (source.z - reciecer.z);
        temp_points.y = (source.z * reciecer.y - source.y * reciecer.z + (p.z + 0.5 * D_geo.dz) * source.y - (p.z + 0.5 * D_geo.dz) * reciecer.y) / (source.z - reciecer.z);
        temp_points.z = p.z + 0.5 * D_geo.dz;
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.z * reciecer.x - source.x * reciecer.z + (p.z - 0.5 * D_geo.dz) * source.x - (p.z - 0.5 * D_geo.dz) * reciecer.x) / (source.z - reciecer.z);
        temp_points.y = (source.z * reciecer.y - source.y * reciecer.z + (p.z - 0.5 * D_geo.dz) * source.y - (p.z - 0.5 * D_geo.dz) * reciecer.y) / (source.z - reciecer.z);
        temp_points.z = p.z - 0.5 * D_geo.dz;
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }
        if (cross_points_2 == Point_3D(-1, -1, -1) && cross_points_1 != Point_3D(-1, -1, -1))
        {
            cross_points_1 = Point_3D(-1, -1, -1);
        }

        D_weight[index] = (cross_points_1 - cross_points_2).norm();

    }
}

__global__ void Par_CalcWeight_kernel(float* D_weight, Point_3D *p_source, Point_3D *p_reciecer, unsigned long num_rays) {

    unsigned long Dataidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long Projidx = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned long Xidx = (Dataidx % (D_geo.nx * D_geo.ny)) % D_geo.nx;
    unsigned long Yidx = (Dataidx % (D_geo.nx * D_geo.ny)) / D_geo.nx;
    unsigned long Zidx = Dataidx / (D_geo.nx * D_geo.ny);    

    if (Dataidx >= (D_geo.nx * D_geo.ny * D_geo.nz) || Projidx > num_rays)
    {
        return;
    }
    Point_3D source = p_source[Projidx];
    Point_3D reciecer = p_reciecer[Projidx];

    // 扫描收集射线经过的像素
    // 设定约束：Point(i,j,k)至射线距离小于半个像素）且不超出端点范围
    Point_3D p = Point_3D(Xidx * D_geo.dx, Yidx * D_geo.dy, Zidx * D_geo.dz);
    float length = (reciecer - source).norm();
    float distance = cross(reciecer - source, p - source).norm() / length;
    //if (Xidx == 12 && Zidx == 12) {
    //    printf("(%f, %f, %f, %f, %f, %f)\n", p.x, p.y, p.z, length, distance, (Point_3D(D_geo.dx, D_geo.dy, D_geo.dz).norm() / 2));
    //}

    Point_3D temp_points = Point_3D(-1, -1, -1);
    Point_3D cross_points_1 = Point_3D(-1, -1, -1);
    Point_3D cross_points_2 = Point_3D(-1, -1, -1);

    if ((distance < (Point_3D(D_geo.dx, D_geo.dy, D_geo.dz).norm() / 2))
        && ((p - source).norm() <= length)
        && ((p - reciecer).norm() <= length)) {

        // 求射线与所经过路径的体像素交叉点
        temp_points.x = p.x + 0.5 * D_geo.dx;
        temp_points.y = (source.x * reciecer.y - source.y * reciecer.x + (p.x + 0.5 * D_geo.dx) * source.y - (p.x + 0.5 * D_geo.dx) * reciecer.y) / (source.x - reciecer.x);
        temp_points.z = (source.x * reciecer.z - source.z * reciecer.x + (p.x + 0.5 * D_geo.dx) * source.z - (p.x + 0.5 * D_geo.dx) * reciecer.z) / (source.x - reciecer.x);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = p.x - 0.5 * D_geo.dx;
        temp_points.y = (source.x * reciecer.y - source.y * reciecer.x + (p.x - 0.5 * D_geo.dx) * source.y - (p.x - 0.5 * D_geo.dx) * reciecer.y) / (source.x - reciecer.x);
        temp_points.z = (source.x * reciecer.z - source.z * reciecer.x + (p.x - 0.5 * D_geo.dx) * source.z - (p.x - 0.5 * D_geo.dx) * reciecer.z) / (source.x - reciecer.x);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.y * reciecer.x - source.x * reciecer.y + (p.y + 0.5 * D_geo.dy) * source.x - (p.y + 0.5 * D_geo.dy) * reciecer.x) / (source.y - reciecer.y);
        temp_points.y = p.y + 0.5 * D_geo.dy;
        temp_points.z = (source.y * reciecer.z - source.z * reciecer.y + (p.y + 0.5 * D_geo.dy) * source.z - (p.y + 0.5 * D_geo.dy) * reciecer.z) / (source.y - reciecer.y);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.y * reciecer.x - source.x * reciecer.y + (p.y - 0.5 * D_geo.dy) * source.x - (p.y - 0.5 * D_geo.dy) * reciecer.x) / (source.y - reciecer.y);
        temp_points.y = p.y - 0.5 * D_geo.dy;
        temp_points.z = (source.y * reciecer.z - source.z * reciecer.y + (p.y - 0.5 * D_geo.dy) * source.z - (p.y - 0.5 * D_geo.dy) * reciecer.z) / (source.y - reciecer.y);
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.z * reciecer.x - source.x * reciecer.z + (p.z + 0.5 * D_geo.dz) * source.x - (p.z + 0.5 * D_geo.dz) * reciecer.x) / (source.z - reciecer.z);
        temp_points.y = (source.z * reciecer.y - source.y * reciecer.z + (p.z + 0.5 * D_geo.dz) * source.y - (p.z + 0.5 * D_geo.dz) * reciecer.y) / (source.z - reciecer.z);
        temp_points.z = p.z + 0.5 * D_geo.dz;
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }

        temp_points.x = (source.z * reciecer.x - source.x * reciecer.z + (p.z - 0.5 * D_geo.dz) * source.x - (p.z - 0.5 * D_geo.dz) * reciecer.x) / (source.z - reciecer.z);
        temp_points.y = (source.z * reciecer.y - source.y * reciecer.z + (p.z - 0.5 * D_geo.dz) * source.y - (p.z - 0.5 * D_geo.dz) * reciecer.y) / (source.z - reciecer.z);
        temp_points.z = p.z - 0.5 * D_geo.dz;
        // 判断射线与该平面是否交叉，如交叉且未重复则记录
        if ((temp_points.x <= (p.x + 0.5 * D_geo.dx)) && (temp_points.x >= (p.x - 0.5 * D_geo.dx))
            && (temp_points.y <= (p.y + 0.5 * D_geo.dy)) && (temp_points.y >= (p.y - 0.5 * D_geo.dy))
            && (temp_points.z <= (p.z + 0.5 * D_geo.dz)) && (temp_points.z >= (p.z - 0.5 * D_geo.dz)))
        {
            if (cross_points_1 == Point_3D(-1, -1, -1))
            {
                cross_points_1 = temp_points;
            }
            else if (cross_points_2 == Point_3D(-1, -1, -1))
            {
                cross_points_2 = temp_points;
            }
            if (cross_points_1 == cross_points_2)
            {
                cross_points_2 = Point_3D(-1, -1, -1);
            }
        }
        if (cross_points_2 == Point_3D(-1, -1, -1) && cross_points_1 != Point_3D(-1, -1, -1))
        {
            cross_points_1 = Point_3D(-1, -1, -1);
        }

        unsigned long index = Projidx * D_geo.nx * D_geo.ny * D_geo.nz + Dataidx;
        D_weight[index] = (cross_points_1 - cross_points_2).norm();

    }
}
