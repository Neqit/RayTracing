#ifndef RECTANGLE_H
#define RECTANGLE_H

#include "hittable.h"
#include "material.h"
#include "ray.h"
#include "Vector3.h"

class xy_rect : public hittable
{
public:
	__device__ xy_rect() {}
	__device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat)
		: x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

public:
	material* mp;
	float x0, x1, y0, y1, k;
};

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	float t = (k - r.origin().z()) / r.direction().z();
	if (t < t_min || t > t_max)
		return false;
	float x = r.origin().x() + t * r.direction().x();
	float y = r.origin().y() + t * r.direction().y();
	if (x < x0 || x > x1 || y < y0 || y > y1)
		return false;
	//rec.u = (x - x0) / (x1 - x0);
	//rec.v = (y - y0) / (y1 - y0);
	//rec.t = t;
	Vector3 outward_normal = Vector3(0, 0, 1);
	//rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;
}



#endif // !RECTANGLE_H
