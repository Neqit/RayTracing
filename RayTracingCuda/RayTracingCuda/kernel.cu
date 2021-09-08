#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <iostream>
#include <time.h>

#include "Vector3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "rectangle.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}



__device__ Vector3 ray_color(const ray& r, hittable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    Vector3 cur_attenuation = Vector3(1.0, 1.0, 1.0);
    //max depth = 50
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            Vector3 attenuation;
            Vector3 emitted = rec.mat_ptr->emitted(); //NEW

            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation =  emitted + cur_attenuation * attenuation; //NEW
                cur_ray = scattered;
            }
            else {
                return emitted * cur_attenuation;//Vector3(0.0, 0.0, 0.0); //return emmited //NEW
            }
        }
        else {
            //global illumination
            /*Vector3 unit_direction = normalize(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vector3 c = (1.0f - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
            return cur_attenuation * c;*/
            return Vector3(0.0, 0.0, 0.0);
        }
    }

    return Vector3(0.0, 0.0, 0.0); //exceed depth
}

__global__ void render(Vector3* fb, int max_x, int max_y, int ns, camera** cam, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; //Using the threadIdx and blockIdx CUDA built-in variables we identify the coordinates of each thread in the image (i,j)
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    curandState local_rand_state = rand_state[pixel_index];
    Vector3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    //sqrt = gamma correction
    col[0] = std::sqrt(col[0]);
    col[1] = std::sqrt(col[1]);
    col[2] = std::sqrt(col[2]);

    fb[pixel_index] = col;
}

//separate kernel for time measure
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread will receive same seed = same starting states
    curand_init(2021, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(Vector3(0, -100, -1), 100, new lambertian(Vector3(0.8, 0.8, 0.0))); //ground
        d_list[1] = new sphere(Vector3(0, 0.5, 0), 0.5, new metal(Vector3(0.7, 0.6, 0.5), 0.1));
        d_list[2] = new sphere(Vector3(-1, 0.5, 0), 0.5, new lambertian(Vector3(0.3, 0.0, 0.9)));
        d_list[3] = new sphere(Vector3(1, 0.5, 0), 0.5, new metal(Vector3(0.7, 0.6, 0.5), 0.95));
        d_list[4] = new sphere(Vector3(-2, 0.5, 0), 0.5, new metal(Vector3(0.7, 0.6, 0.5), 0.6));
        d_list[5] = new sphere(Vector3(2, 0.5, 0), 0.5, new metal(Vector3(0.7, 0.6, 0.5), 0.3));
        d_list[6] = new sphere(Vector3(3, 0.5, 0), 0.5, new metal(Vector3(0.7, 0.6, 0.5), 0.02));
        d_list[7] = new sphere(Vector3(0, 1.5, -2), 1.5, new metal(Vector3(0.7, 0.6, 0.5), 0.1));
        d_list[8] = new sphere(Vector3(0.15, 0.2, 1), 0.2, new lambertian(Vector3(0.7, 0.0, 0.99)));
        d_list[9] = new sphere(Vector3(0.8, 0.2, 0.9), 0.2, new lambertian(Vector3(0.6, 0.3, 0.9)));
        d_list[10] = new sphere(Vector3(2.5, 0.2, 1), 0.2, new lambertian(Vector3(0.9, 0.99, 0.1)));
        //d_list[11] = new sphere(Vector3(0.5, 0.2, 2), 0.2, new lambertian(Vector3(0.99, 0.0, 0.2)));
        d_list[11] = new sphere(Vector3(0.5, 0.2, 2), 0.2, new diffuse_light(Vector3(1, 0.0, 0.0)));
        d_list[12] = new sphere(Vector3(-0.2, 0.2, 0.77), 0.2, new lambertian(Vector3(0.7, 0.99, 0.0)));


        d_list[13] = new xy_rect(-10, 5, -3, 3, -3, new diffuse_light(Vector3(1, 1, 1)));

        *d_world = new hittable_list(d_list, 14);
        Vector3 lookfrom(5, 2, 5);
        Vector3 lookat(0, 0, -1);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            Vector3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}


__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i = 0; i < 13; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }

    delete ((xy_rect*)d_list[13])->mp; 
    delete d_list[13];

    delete* d_world;
    delete* d_camera;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int ns = 1000; //samples
    int tx = 8; //threadX
    int ty = 8; //threadY

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny; //frame buffer - has a pixel count that is a multiple of 32 in order to fit into warps evenly.
    size_t fb_size = 3 * num_pixels * sizeof(Vector3); //framebuffer multiplied with 3 channels

    // allocate FB
    Vector3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size)); //memory allocated for all vectors

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));


    // make our world of hitables
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 14 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<<1, 1 >>>(d_list, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>> (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    uint8_t* pixels = new uint8_t[nx * ny * 3];

    int index = 0;

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].x());
            int ig = int(255.99 * fb[pixel_index].y());
            int ib = int(255.99 * fb[pixel_index].z());
            //std::cout << ir << " " << ig << " " << ib << "\n";

            pixels[index++] = ir;
            pixels[index++] = ig;
            pixels[index++] = ib;

        }
    }

    stbi_write_png("cuda_final_light_red.png", nx, ny, 3, pixels, nx * 3);
    delete[] pixels;

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<< 1, 1 >>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}