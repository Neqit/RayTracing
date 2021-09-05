#include <iostream>
#include "Vector3.h"
#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#include <cmath>
#define _USE_MATH_DEFINES
#include <limits>
#include <memory>


#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
const double infinity = std::numeric_limits<double>::infinity();
//const double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}



color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    //Maximum recursion depth. It will stop when there is nothing to hit
    if (depth <= 0) {
        return color(0, 0, 0);
    }

    //check for hitting an object
    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return color(0, 0, 0);
    }

    //Otherwise returns blue gradient (sky)
    Vector3 unit_direction = normalize(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 640;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int chanel_num = 3;
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    //Camera
    camera cam;

    //World
    hittable_list world;

    auto material_ground = std::make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<lambertian>(color(0.7, 0.3, 0.3));
    auto material_left = std::make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
    auto material_right = std::make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

    world.add(std::make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(std::make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(std::make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    // Render
    uint8_t* pixels = new uint8_t[image_width * image_height * chanel_num];

    int index = 0;

    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {

            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }

            color pixels_color = write_color(std::cout, pixel_color, samples_per_pixel);

            pixels[index++] = static_cast<int>(pixels_color.x());
            pixels[index++] = static_cast<int>(pixels_color.y());
            pixels[index++] = static_cast<int>(pixels_color.z());
        }
    }

    stbi_write_png("fuzz.png", image_width, image_height, chanel_num, pixels, image_width * chanel_num);
    delete[] pixels;

}