#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "Vector3.h"

class camera
{
public:
	camera() {
		auto aspect_ratio = 16.0 / 9.0;
		auto viewport_height = 2.0;
		auto viewport_width = aspect_ratio * viewport_height;
		auto focal_length = 1.0;

		origin = point3(0, 0, 0);
		horizontal = Vector3(viewport_width, 0.0, 0.0);
		vertical = Vector3(0.0, viewport_height, 0.0);
		lowr_left_corner = origin - horizontal / 2 - vertical / 2 - Vector3(0, 0, focal_length);
	};

	ray get_ray(double u, double v) const {
		return ray(origin, lowr_left_corner + u * horizontal + v * vertical - origin);
	}

private:
	point3 origin;
	point3 lowr_left_corner;
	Vector3 horizontal;
	Vector3 vertical;

};

#endif // !CAMERA_H
