#ifndef CAMERA_H
#define CAMERA_H

#define _USE_MATH_DEFINES

#include "ray.h"
#include "Vector3.h"
#include <curand_kernel.h>


#include <cmath>


__device__ Vector3 random_in_unit_disk(curandState* local_rand_state) {
	Vector3 p;
	do {
		p = 2.0f * Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vector3(1, 1, 0);
	} while (dot(p, p) >= 1.0f);
	return p;
}

class camera
{
public:
    __device__ camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist) {
        lens_radius = aperture / 2.0f;
        float theta = vfov * ((float)3.14159265358979323846f) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);
        lowr_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }
    __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
        Vector3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        Vector3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lowr_left_corner + s * horizontal + t * vertical - origin - offset);
    }

private:
    Vector3 origin;
    Vector3 lowr_left_corner;
    Vector3 horizontal;
    Vector3 vertical;
    Vector3 u, v, w;
    float lens_radius;
};

#endif // !CAMERA_H
