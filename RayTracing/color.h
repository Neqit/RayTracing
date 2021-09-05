#ifndef COLOR_H
#define COLOR_H

#include "Vector3.h"
#include <iostream>

inline double clamp(double x, double min, double max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

//translates 8bit value of each pixel color (multisampled)
color write_color(std::ostream& out, color pixel_color, int samples_per_pixel) {
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	//sampling multiple colors with gamma correction 2.0
	auto scale = 1.0 / samples_per_pixel;
	r = sqrt(scale * r);
	g = sqrt(scale * g);
	b = sqrt(scale * b);


	//write translation
	/*out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';*/

	return color(256 * clamp(r, 0.0, 0.999), 256 * clamp(g, 0.0, 0.999), 256 * clamp(b, 0.0, 0.999));

}

#endif // !COLOR

