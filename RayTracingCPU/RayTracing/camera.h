#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "Vector3.h"


inline double degrees_to_radians(double degrees) {
	return degrees * 3.1415926535897932385 / 180.0;
}

class camera
{
public:
	camera(point3 lookfrom, point3 lookat, Vector3 view_up, double vfov, double aspect_ratio, double aperture, double focus_dist) {
		auto theta = degrees_to_radians(vfov);
		auto h = tan(theta / 2);
		auto viewport_height = 2.0 * h;
		auto viewport_width = aspect_ratio * viewport_height;
		auto focal_length = 1.0;

		 w = normalize(lookfrom - lookat);
		 u = normalize(cross(view_up, w));
		 v = cross(w, u);


		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lowr_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

		lens_radius = aperture / 2;
	};

	ray get_ray(double s, double t) const {
		Vector3 rd = lens_radius * random_in_unit_disk();
		Vector3 offset = u * rd.x() + v * rd.y();

		return ray(origin + offset, lowr_left_corner + s * horizontal + t * vertical - origin - offset);
	}

private:
	point3 origin;
	point3 lowr_left_corner;
	Vector3 horizontal;
	Vector3 vertical;
	Vector3 u, v, w;
	double lens_radius;

};

#endif // !CAMERA_H
