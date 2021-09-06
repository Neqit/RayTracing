
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>

#include "Vector3.h"
#include "ray.h"

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

__device__ Vector3 ray_color(const ray& r) {
    Vector3 unit_direction = normalize(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f); //force single precision with floats
    return (1.0f - t) * Vector3(1.0, 1.0, 1.0) = t * Vector3(0.5, 0.7, 1.0);
}

__global__ void render(Vector3* fb, int max_x, int max_y, Vector3 lower_left_corner, Vector3 horizontal, Vector3 vertical, Vector3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; //Using the threadIdx and blockIdx CUDA built-in variables we identify the coordinates of each thread in the image (i,j)
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner +u * horizontal + v * vertical);

    fb[pixel_index] = ray_color(r);
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8; //threadX
    int ty = 8; //threadY

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny; //frame buffer - has a pixel count that is a multiple of 32 in order to fit into warps evenly.
    size_t fb_size = 3 * num_pixels * sizeof(Vector3); //framebuffer multiplied with 3 channels

    // allocate FB
    Vector3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>>(fb, nx, ny, Vector3(-2.0, -1.0, -1.0), Vector3(4.0,0.0,0.0), Vector3(0.0,2.0,0.0), Vector3(0.0,0.0,0.0));
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

    stbi_write_png("cuda.png", nx, ny, 3, pixels, nx * 3);
    delete[] pixels;
    checkCudaErrors(cudaFree(fb));
}