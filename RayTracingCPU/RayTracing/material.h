#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "Vector3.h"
//#include "texture.h"

struct hit_record;

class material
{
public:

    virtual color emitted() const {
        return color(0, 0, 0);
    }

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;

};

class lambertian : public material {
public:
    lambertian(const color& a) : albedo(a) {}
    //lambertian(std::shared_ptr<texture> a) : albedo(a) {}

    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        auto scatter_direction = rec.normal + random_unit_vector();

        //Zero scatter dir
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    color albedo;
};


class metal : public material {
public:
    metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        Vector3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

public:
    color albedo;
    double fuzz;
};


class diffuse_light : public material {
public:
    //diffuse_light(std::shared_ptr<color> a) : emit(a) {}
    diffuse_light(color c) : emit(c) {}
    
    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        return false;
    }

    virtual color emitted() const override {
        return emit;
    }


public:
    color emit;
};


#endif // !MATERIAL_H
