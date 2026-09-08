#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>

#include "camera.hpp"
#include "gaussian.hpp"
#include "raster.hpp"
#include "utils.hpp"
#include "rmm/device_uvector.hpp"

void savePPM(const std::string& filename, float* image, int width, int height) {
    std::ofstream f(filename);
    f << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; i++)
        f << (int)(std::min(1.0f, image[i]) * 255) << " ";
}

static void rasterization_bench(benchmark::State& state)
{
    std::string path = "/home/h/Downloads/cuSplat/point_cloud.ply";
    size_t length = getSize(path);
    Gaussian* gaussians = nullptr;
    cudaError_t err = cudaHostAlloc((void**)&gaussians, length * sizeof(Gaussian), 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << "\n";
        return;
    }
    loadPLY(path, gaussians);
    int width, height;
    width = 1959;
    height = 1090;
    float pos[3] = {-2.5199776022057296f, -0.09704735754873686f, -3.6247725540304545f};
    float R[3][3] = {
        { 0.9982731285632193f,  -0.011928707708098955f, -0.05751927260507243f},
        { 0.0065061360949636f,   0.9955928229282383f,   -0.09355533724430458f},
        { 0.0583817692581828f,   0.09301955098900708f,   0.9939511719154457f }
    };
    Camera cam(pos, R, width, height, 1159.5880733038064f, 1164.6601287484507f, 979.5f, 545.0f);
    float *image = nullptr;
    err = cudaHostAlloc((void**)&image, width * height * 3 * sizeof(float), 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << "\n";
        return;
    }
    auto gaussiansMemory = rmm::device_uvector<Gaussian>(length, rmm::cuda_stream_default);
    Gaussian* d_raw = gaussiansMemory.data();
    cudaMemcpyAsync(d_raw, gaussians, length*sizeof(Gaussian), cudaMemcpyHostToDevice, rmm::cuda_stream_default);
    Raster rast = Raster(raft::device_span<Gaussian>(d_raw, length), cam);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (auto _ : state) {
        cudaEventRecord(start, rmm::cuda_stream_default);
        rast.rasterize(image);
        cudaEventRecord(stop, rmm::cuda_stream_default);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);

    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    savePPM("/home/h/Downloads/cuSplat/out/output.ppm", image, width, height);
    cudaFreeHost(image);
    cudaFreeHost(gaussians);

}
BENCHMARK(rasterization_bench)->Unit(benchmark::kSecond)->UseManualTime()->Iterations(1);
BENCHMARK(rasterization_bench)->Unit(benchmark::kSecond)->UseManualTime()->Iterations(200);

BENCHMARK_MAIN();
