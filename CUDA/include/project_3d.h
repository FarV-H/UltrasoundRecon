#pragma once

#ifndef PROJECT_3D_FILE
#define PROJECT_3D_FILE

#include "types.h"

__host__ __device__ float* Project3D(Point_3D* source,
    Point_3D* reciecer,
    float* projection,
    float* data,
    Geometry geo);

#endif // !PROJECT_3D_FILE


