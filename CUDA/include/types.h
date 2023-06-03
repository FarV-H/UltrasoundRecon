#pragma once

#ifndef TYPES_FILE
#define TYPES_FILE

#include "cuda_utils.h"
#include "math.h"

class Point_3D {
public:
	__host__ __device__ Point_3D::Point_3D(float a, float b, float c) {
		this->x = a;
		this->y = b;
		this->z = c;
	}
	__host__ __device__ bool operator==(Point_3D& p) {
		if (this->x == p.x && this->y == p.y && this->z == p.z) {
			return true;
		}
		return false;
	}
	__host__ __device__ bool operator!=(Point_3D& p) {
		if (this->x == p.x && this->y == p.y && this->z == p.z) {
			return false;
		}
		return true;
	}
	float x;
	float y;
	float z;
	__host__ __device__ float norm() {
		return sqrt(x * x + y * y + z * z);
	}
};

__host__ __device__ Point_3D operator +(const Point_3D& a1, const Point_3D& a2);
__host__ __device__ Point_3D operator -(const Point_3D& a1, const Point_3D& a2);
//__host__ __device__ Point_3D operator *(const Point_3D& a1, const Point_3D& a2);
//__host__ __device__ Point_3D operator /(const Point_3D& a1, const Point_3D& a2);
__host__ __device__ Point_3D cross(const Point_3D& a1, const Point_3D& a2);

struct Geometry {
	unsigned long nx;
	unsigned long ny;
	unsigned long nz;
	float sx;
	float sy;
	float sz;
	float dx;
	float dy;
	float dz;
	unsigned long num_channels_x;
	unsigned long num_channels_y;
	unsigned long num_channels;
	unsigned long num_views;
	unsigned long num_panel;
	unsigned long interp;
};

#endif // !TYPES_FILE

