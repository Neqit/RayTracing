#include <iostream>
#include "Vector3.h"
#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "rectangle.h"

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


color ray_color(const ray& r, const color& background, const hittable& world, int depth) {
    hit_record rec;

    //Maximum recursion depth. It will stop when there is nothing to hit
    if (depth <= 0) {
        return color(0, 0, 0);
    }

    // If the ray hits nothing, return the background color.
    if (!world.hit(r, 0.001, infinity, rec)) {
        //return background;
        //Otherwise returns blue gradient (sky)
        Vector3 unit_direction = normalize(r.direction());
        auto t = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
    }


    //check for hitting an object
    ray scattered;
    color attenuation;
    color emitted = rec.mat_ptr->emitted();

    if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
        return emitted;

    return emitted + attenuation * ray_color(scattered, background, world, depth - 1);
}

void render(int image_height, int image_width, int samples_per_pixel, camera& cam, color background, const hittable& world, int max_depth, uint8_t* pixels, int& index) {
    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {

            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, background, world, max_depth);
            }

            color pixels_color = write_color(std::cout, pixel_color, samples_per_pixel);

            pixels[index++] = static_cast<int>(pixels_color.x());
            pixels[index++] = static_cast<int>(pixels_color.y());
            pixels[index++] = static_cast<int>(pixels_color.z());
        }
    }
}

int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 200;
    const int image_height = 100;//static_cast<int>(image_width / aspect_ratio);
    const int chanel_num = 3;
    const int samples_per_pixel = 20;
    const int max_depth = 50;

    //Camera
    point3 lookfrom(5, 2, 5);
    point3 lookat(0, 0, -1);
    Vector3 view_up(0, 1, 0);
    auto dist_to_focus = (lookfrom - lookat).length();
    auto aperture = 0.1;
    auto vertical_fov = 30;
    camera cam(lookfrom, lookat, view_up, vertical_fov, aspect_ratio, aperture, dist_to_focus);
    color background(0, 0, 0);

    //World
    //auto R = cos(M_PI / 4);
    /*hittable_list world;

    auto material_ground = std::make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<lambertian>(color(0.7, 0.3, 0.3));
    auto material_left = std::make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
    auto material_right = std::make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

    world.add(std::make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(std::make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(std::make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));*/

    hittable_list world;


        /*auto material_ground = std::make_shared<lambertian>(color(0.8, 0.8, 0.0));
        auto material_center = std::make_shared<metal>(color(0.7, 0.3, 0.3),0.5);
        world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, material_ground));
        world.add(std::make_shared<sphere>(point3(0, 2, 0), 2, material_center));*/

       // auto difflight = std::make_shared<diffuse_light>(color(4, 4, 4));
       // world.add(std::make_shared<xy_rect>(3, 5, 1, 3, -2, difflight));

        //world.add(std::make_shared<sphere>(point3(0, 10, 0), 2, difflight));

        world.add(std::make_shared<sphere>(Vector3(0, -100, -1), 100, std::make_shared<lambertian>(Vector3(0.8, 0.8, 0.0)))); //ground
        world.add(std::make_shared<sphere>(Vector3(0, 0.5, 0), 0.5, std::make_shared<metal>(Vector3(0.7, 0.6, 0.5), 0.1)));
        world.add(std::make_shared<sphere>(Vector3(-1, 0.5, 0), 0.5, std::make_shared<lambertian>(Vector3(0.3, 0.0, 0.9))));
        world.add(std::make_shared<sphere>(Vector3(1, 0.5, 0), 0.5, std::make_shared<metal>(Vector3(0.7, 0.6, 0.5), 0.95)));
        world.add(std::make_shared<sphere>(Vector3(-2, 0.5, 0), 0.5, std::make_shared<metal>(Vector3(0.7, 0.6, 0.5), 0.6)));
        world.add(std::make_shared<sphere>(Vector3(2, 0.5, 0), 0.5, std::make_shared<metal>(Vector3(0.7, 0.6, 0.5), 0.3)));
        world.add(std::make_shared<sphere>(Vector3(3, 0.5, 0), 0.5, std::make_shared<metal>(Vector3(0.7, 0.6, 0.5), 0.02)));
        world.add(std::make_shared<sphere>(Vector3(0, 1.5, -2), 1.5, std::make_shared<metal>(Vector3(0.7, 0.6, 0.5), 0.1)));
        world.add(std::make_shared<sphere>(Vector3(0.15, 0.2, 1), 0.2, std::make_shared<lambertian>(Vector3(0.7, 0.0, 0.99))));
        world.add(std::make_shared<sphere>(Vector3(0.8, 0.2, 0.9), 0.2, std::make_shared<lambertian>(Vector3(0.6, 0.3, 0.9))));
        world.add(std::make_shared<sphere>(Vector3(2.5, 0.2, 1), 0.2, std::make_shared<lambertian>(Vector3(0.9, 0.99, 0.1))));
        //d_list[11] = new sphere(Vector3(0.5, 0.2, 2), 0.2, new lambertian(Vector3(0.99, 0.0, 0.2)));
        world.add(std::make_shared<sphere>(Vector3(0.5, 0.2, 2), 0.2, std::make_shared<lambertian>(Vector3(0.99, 0.0, 0.2))));
        world.add(std::make_shared<sphere>(Vector3(-0.2, 0.2, 0.77), 0.2, std::make_shared<lambertian>(Vector3(0.7, 0.99, 0.0))));


        //world.add(std::make_shared<xy_rect>(-10, 5, -3, 3, -3, new diffuse_light(Vector3(1, 1, 1))));
    

    // Render
    uint8_t* pixels = new uint8_t[image_width * image_height * chanel_num];

    int index = 0;

    render(image_height, image_width, samples_per_pixel, cam, background, world, max_depth, pixels, index);

    stbi_write_png("cpu_final.png", image_width, image_height, chanel_num, pixels, image_width * chanel_num);
    delete[] pixels;

}