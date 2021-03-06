#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

class material;

struct hit_record
{
	point3 p;
	Vector3 normal;
	material* mat_ptr;
	float t;


	/*inline void set_face_normal(const ray& r, const Vector3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}*/
};

class hittable
{
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

};


#endif // !HITTABLE_H
