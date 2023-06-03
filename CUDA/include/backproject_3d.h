#pragma once

#ifndef BACKPROJECT_3D_FILE
#define BACKPROJECT_3D_FILE

#include "types.h"

__host__ __device__ float* Backproject3D(Point_3D* source,
    Point_3D* reciecer,
    float* projection,
    float* data,
    Geometry geo);

#endif // !BACKPROJECT_3D_FILE


