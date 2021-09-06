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

#define RANDVEC3 Vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vector3 random_in_unit_sphere(curandState* local_rand_state) {
    Vector3 p;
    do {
        p = 2.0f * RANDVEC3 - Vector3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}


__device__ Vector3 ray_color(const ray& r, hittable** world, curandState *local_rand_state) {
    ray cur_ray = r;
    float cur_attenuation = 1.0f;
    //max depth = 50
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Vector3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f;
            cur_ray = ray(rec.p, target - rec.p);
        }
        else {
            Vector3 unit_direction = normalize(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vector3 c = (1.0f - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
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
        ray r = (*cam)->get_ray(u, v);
        col += ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
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
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(Vector3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(Vector3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
        *d_camera = new camera();
    }
}


__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
    delete* d_camera;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int ns = 100; //samples
    int tx = 8; //threadX
    int ty = 8; //threadY

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny; //frame buffer - has a pixel count that is a multiple of 32 in order to fit into warps evenly.
    size_t fb_size = 3 * num_pixels * sizeof(Vector3); //framebuffer multiplied with 3 channels

    // allocate FB
    Vector3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));


    // make our world of hitables
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
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

    stbi_write_png("cuda_diffuse.png", nx, ny, 3, pixels, nx * 3);
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