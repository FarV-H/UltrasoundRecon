#include "types.h"
#include "cuda_utils.h"

__host__ __device__ Point_3D operator +(const Point_3D& a1, const Point_3D& a2)
{
	Point_3D tmp(a1);
	tmp.x += a2.x;
	tmp.y += a2.y;
	tmp.z += a2.z;
	return tmp;
}
__host__ __device__ Point_3D operator -(const Point_3D& a1, const Point_3D& a2)
{
	Point_3D tmp(a1);
	tmp.x -= a2.x;
	tmp.y -= a2.y;
	tmp.z -= a2.z;
	return tmp;
}
//__host__ __device__ Point_3D operator *(const Point_3D& a1, const Point_3D& a2)
//{
//	Point_3D tmp(a1);
//	tmp.x *= a2.x;
//	tmp.y *= a2.y;
//	tmp.z *= a2.z;
//	return tmp;
//}
//__host__ __device__ Point_3D operator /(const Point_3D& a1, const Point_3D& a2)
//{
//	Point_3D tmp(a1);
//	tmp.x /= a2.x;
//	tmp.y /= a2.y;
//	tmp.z /= a2.z;
//	return tmp;
//}
__host__ __device__ Point_3D cross(const Point_3D& a1, const Point_3D& a2)
{
	Point_3D tmp(a1);
	tmp.x = a1.y * a2.z - a1.z * a2.y;
	tmp.y = a1.z * a2.x - a1.x * a2.z;
	tmp.z = a1.x * a2.y - a1.y * a2.x;
	return tmp;
}
