#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "Vector3.h"
#include <curand_kernel.h>
#include "hittable.h"

struct hit_record;

#define RANDVEC3 Vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vector3 random_in_unit_sphere(curandState* local_rand_state) {
    Vector3 p;
    do {
        p = 2.0f * RANDVEC3 - Vector3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ Vector3 reflect(const Vector3& v, const Vector3& n) {
    return v - 2.0f * dot(v, n) * n;
}


class material
{
public:

    __device__ virtual color emitted() const {
        return color(0, 0, 0);
    }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const = 0;

};

class lambertian : public material {
public:
    __device__ lambertian(const Vector3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, Vector3& attenuation, ray& scattered, curandState* local_rand_state) const {
        Vector3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

private:
    Vector3 albedo;
};


class metal : public material {
public:
    __device__ metal(const Vector3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, Vector3& attenuation, ray& scattered, curandState* local_rand_state) const {
        Vector3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

private:
    Vector3 albedo;
    float fuzz;
};


class diffuse_light : public material {
public:
    __device__ diffuse_light(color c) : emit(c) {}
    
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const override {
        return false;
    }

    __device__ virtual color emitted() const override {
        return emit;
    }


public:
    color emit;
};


#endif // !MATERIAL_H
