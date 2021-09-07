#ifndef TEXTURE_H
#define TEXTURE_H

#include "ray.h"

/*class texture
{
public:
	__device__ virtual color value(float u, float v, const point3& p) const = 0;
};


class solid_color : public texture
{
public:
	__device__ solid_color() {}
	__device__ solid_color(color c) : color_value(c) {}

	__device__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

	__device__ virtual color value(float u, float v, const point3& p) const override {
		return color_value;
	}


private:
	color color_value;
};*/


#endif // !TEXTURE_H
