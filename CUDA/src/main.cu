#include "types.h"
#include "project_3d.h"
#include "backproject_3d.h"

int main() {

	float* projection = (float*)malloc(64 * 32 * sizeof(float));
	float* data = (float*)malloc(128 * 128 * 128 * sizeof(float));

	for (size_t i = 0; i < 32 * 64; i++)
	{
		projection[i] = 1;
	}
	for (size_t i = 0; i < 128 * 128 * 128; i++)
	{
		data[i] = 1;
	}

	Point_3D* source = (Point_3D*)malloc(32 * 2 * sizeof(Point_3D));
	Point_3D* reciecer = (Point_3D*)malloc(32 * 2 * sizeof(Point_3D));

	Geometry geo;
	geo.nx = 128;
	geo.ny = 128;
	geo.nz = 128;
	geo.sx = 500.0;
	geo.sy = 500.0;
	geo.sz = 500.0;
	geo.dx = geo.sx / geo.nx;
	geo.dy = geo.sy / geo.ny;
	geo.dz = geo.sz / geo.nz;
	geo.num_views = 32;
	geo.num_channels_x = 4;
	geo.num_channels_y = 8;
	geo.num_channels = 32;
	geo.num_panel = 2;

	for (size_t i = 0; i < geo.num_views * geo.num_panel; i++)
	{
		source[i] = { Point_3D(50, 0, 50) };
		reciecer[i] = { Point_3D(50, 500, 50) };
	}



	Backproject3D(source, reciecer, projection, data, geo);
}