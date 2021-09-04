#ifndef COLOR_H
#define COLOR_H

#include "Vector3.h"
#include <iostream>

//translates 8bit value of each pixel color
void write_color(std::ostream& out, color pixel_color) {
	out << static_cast<int>(255.99 * pixel_color.x()) << ' '
		<< static_cast<int>(255.99 * pixel_color.y()) << ' '
		<< static_cast<int>(255.99 * pixel_color.z()) << '\n';
}

#endif // !COLOR

