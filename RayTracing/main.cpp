#include <iostream>
#include "Vector3.h"
#include "color.h"
#include "ray.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//quadratic equation to check if ray hits the sphere  
// basicly returns true if disciminant > 0
// (0 roots = 0 intersextions, 1 root = 1 intersection, 2 roots = 2 intersections)
bool hit_sphere(const point3& center, double radius, const ray& r) {
    Vector3 oc = r.origin() - center; //vector from center to point
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;
    return (discriminant > 0);
}

//blue gradient image
color ray_color(const ray& r) {
    //draw a red circle
    if (hit_sphere(point3(0, 0, -1), 0.5, r))
        return color(0, 1, 0);

    Vector3 unit_direction = normalize(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * (color(0.5, 0.7, 1.0)); //blend value
}

int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int chanel_num = 3;

    //Camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = Vector3(viewport_width, 0, 0);
    auto vertical = Vector3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - Vector3(0, 0, focal_length);

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    uint8_t* pixels = new uint8_t[image_width * image_height * chanel_num];

    int index = 0;

    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            /*auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);*/

            //color pixel_color(double(i) / (image_width - 1), double(j) / (image_height - 1), 0.25);
            //write_color(std::cout, pixel_color);

            auto u = double(i) / (image_width - 1);
            auto v = double(j) / (image_height - 1);
            ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
            color pixel_color = ray_color(r);


            pixels[index++] = static_cast<int>(255.99 *pixel_color.x());
            pixels[index++] = static_cast<int>(255.99 * pixel_color.y());
            pixels[index++] = static_cast<int>(255.99 * pixel_color.z());

            //std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    stbi_write_png("stbpng.png", image_width, image_height, chanel_num, pixels, image_width * chanel_num);
    delete[] pixels;

}