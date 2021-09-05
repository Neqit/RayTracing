#include <iostream>
#include "Vector3.h"
#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

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




//quadratic equation to check if ray hits the sphere  
// basicly returns true if disciminant > 0
// (0 roots = 0 intersextions, 1 root = 1 intersection, 2 roots = 2 intersections)

/*double hit_sphere(const point3& center, double radius, const ray& r) {
    Vector3 oc = r.origin() - center; //vector from center to point
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) {
        return - 1.0;
    }
    else {
        return (-half_b - std::sqrt(discriminant)) / a;
    }
}*/

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    //Maximum recursion depth. It will stop when there is nothing to hit
    if (depth <= 0) {
        return color(0, 0, 0);
    }

    //check for hitting an object
    if (world.hit(r, 0.001, infinity, rec)) {
        point3 target = rec.p + rec.normal + random_unit_vector();
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);
    }

    //Otherwise returns blue gradient (sky)
    Vector3 unit_direction = normalize(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1280;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int chanel_num = 3;
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    //Camera
    /*auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = Vector3(viewport_width, 0, 0);
    auto vertical = Vector3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - Vector3(0, 0, focal_length);*/
    camera cam;

    //World
    hittable_list world;
    world.add(std::make_shared<sphere>(point3(0, 0, -1), 0.5)); //sphere in the center
    world.add(std::make_shared<sphere>(point3(0, -100.5, -1), 100)); //ground;

    // Render

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


            /*auto u = double(i) / (image_width - 1);
            auto v = double(j) / (image_height - 1);
            ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
            color pixel_color = ray_color(r, world);*/
            //write_color(std::cout, pixel_color);

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

    stbi_write_png("gamma_correcton.png", image_width, image_height, chanel_num, pixels, image_width * chanel_num);
    delete[] pixels;

}