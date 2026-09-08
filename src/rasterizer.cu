#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "camera.hpp"
#include "utils.hpp"
#include "gaussian.hpp"
#include "raster.hpp"
#include "tiling.hpp"


__device__ float computeAlpha(const GaussianSplated& g, float px, float py) {
    float dx = px-g.u;
    float dy = py-g.v;
    // sigmaPrime inversion
    float det = g.cov[0][0]*g.cov[1][1] - g.cov[0][1]*g.cov[1][0];
    if (fabsf(det) < 1e-6f) return 0.0f;
    float power = (1/det) * (g.cov[1][1]*dx*dx - 2*g.cov[0][1]*dx*dy + g.cov[0][0]*dy*dy);
    if (power < 0.0f) return 0.0f;
    return g.opacity*expf(-0.5f*power);
}

__global__ void AlphaBlend(const raft::device_span<TileRange> tileRanges, const raft::device_span<GaussianSplated> gaussSplateds,
                           const raft::device_span<uint32_t> gaussianKeysIdx, int width, int height, float* image) {
    int px_global = blockDim.x * blockIdx.x + threadIdx.x;
    int py_global = blockDim.y * blockIdx.y + threadIdx.y;
    /*
    if (py_global >= height || px_global >= width) {
        return;
    }*/
    unsigned int t = blockIdx.x + gridDim.x * blockIdx.y;
    int start = tileRanges[t].start;
    int end = tileRanges[t].end;
    if (start == -1) {
        return;
    }
    float color[3] = {0.0f, 0.0f, 0.0f};
    float T = 1;
    __shared__ GaussianSplated gaussianSplateds1[TILE_SIZE*TILE_SIZE];
    __shared__ GaussianSplated gaussianSplateds2[TILE_SIZE*TILE_SIZE];
    unsigned int length = end - start;
    for (unsigned int gaussian=0; gaussian<length; gaussian+=TILE_SIZE*TILE_SIZE) {
        // collaborative loading by chunk of block's size
        unsigned int tid = threadIdx.x + threadIdx.y*blockDim.x;
        unsigned int gchunkIdx = gaussian / (TILE_SIZE*TILE_SIZE);
        unsigned int offset = (gchunkIdx) * (TILE_SIZE * TILE_SIZE);
        if (tid + offset < length && ! (gchunkIdx & 1)) {
            gaussianSplateds1[tid] = gaussSplateds[gaussianKeysIdx[start + tid + offset]];
        }
        else if (tid + offset < length && gchunkIdx & 1) {
            gaussianSplateds2[tid] = gaussSplateds[gaussianKeysIdx[start + tid + offset]];
        }
        __syncthreads();

        if (T < 0.0001f || py_global >= height || px_global >= width) {
            continue;
        }
        if (! (gchunkIdx & 1)) {
            for (int i = 0; i < TILE_SIZE*TILE_SIZE; i++) {
                float alpha = computeAlpha(gaussianSplateds1[i], px_global, py_global);
                alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
                color[0] += gaussianSplateds1[i].color[0]*T*alpha;
                color[1] += gaussianSplateds1[i].color[1]*T*alpha;
                color[2] += gaussianSplateds1[i].color[2]*T*alpha;
                T *= (1-alpha);
            }
        }
        else {
            for (int i = 0; i < TILE_SIZE*TILE_SIZE; i++) {
                float alpha = computeAlpha(gaussianSplateds2[i], px_global, py_global);
                alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
                color[0] += gaussianSplateds2[i].color[0]*T*alpha;
                color[1] += gaussianSplateds2[i].color[1]*T*alpha;
                color[2] += gaussianSplateds2[i].color[2]*T*alpha;
                T *= (1-alpha);
            }
        }
        /*
        if (T < 0.0001f) {
            break;
        }*/
    }
    if (py_global >= height || px_global >= width) {
        return;
    }
    for (int c = 0; c < 3; c++) {
        image[(py_global*width+px_global)*3+c] = color[c];
    }
}

void Raster::rasterize(float *image) {
    // CULLING
    auto cullGaussiansMemory = rmm::device_uvector<Gaussian>(gaussians.size(), rmm::cuda_stream_default);
    thrust::device_ptr<Gaussian> d_begin(gaussians.data());
    thrust::device_ptr<Gaussian> d_end = d_begin + gaussians.size();
    auto camera_capture = this->camera;
    auto end = thrust::copy_if(d_begin, d_end,
                                    cullGaussiansMemory.begin(), [camera_capture] __device__(Gaussian gaussian) {
        gaussian = worldToCamera(gaussian, camera_capture);
        float radius = camera_capture.fx * max(max(gaussian.sx, gaussian.sy),gaussian.sz) / gaussian.z;
        return isVisible(gaussian, camera_capture, radius);
    });
    cullGaussiansMemory.resize(end-cullGaussiansMemory.begin(), rmm::cuda_stream_default);
    // --

    // PROJECT TO SCREEN SPACE
    auto cullGaussians = raft::device_span<Gaussian>(cullGaussiansMemory.data(), cullGaussiansMemory.size());
    auto cullGaussiansSpMemory = rmm::device_uvector<GaussianSplated>(cullGaussians.size(), rmm::cuda_stream_default);
    auto gaussiansSplateds = raft::device_span<GaussianSplated>(cullGaussiansSpMemory.data(), cullGaussiansSpMemory.size());
    int BS = 1024;
    int GS = (cullGaussians.size() + BS - 1) / BS;
    screenspaceGaussians<<<GS, BS>>>(cullGaussians, camera, gaussiansSplateds);
    // --

    // FILL & SORT GAUSSIANS KEYS
    // FILL
    auto countMemory = rmm::device_scalar<int>(0, rmm::cuda_stream_default);
    GetSize<<<GS, BS>>>(gaussiansSplateds, nTilesX, nTilesY, countMemory.data());
    int nbKeys = countMemory.value(countMemory.stream());
    auto gaussianKeysMemory = rmm::device_uvector<uint64_t>(nbKeys, rmm::cuda_stream_default);
    auto gaussianKeys = raft::device_span<uint64_t>(gaussianKeysMemory.data(), nbKeys);
    auto gaussianKeysIdxMemory = rmm::device_uvector<uint32_t>(nbKeys, rmm::cuda_stream_default);
    auto gaussianKeysIdx = raft::device_span<uint32_t>(gaussianKeysIdxMemory.data(), nbKeys);

    int newVal = 0;
    // use async to avoid CPU stall between GPU set
    countMemory.set_value_async(newVal, rmm::cuda_stream_default);
    FillKeys<<<GS, BS>>>(gaussiansSplateds, nTilesX, nTilesY, gaussianKeys, gaussianKeysIdx, countMemory.data());

    // SORT TODO: Sort gaussianKeys with CUB's radix sort
    thrust::sort_by_key(
                        thrust::cuda::par.on(rmm::cuda_stream_default.value()),
                        thrust::device_ptr<uint64_t>(gaussianKeys.data()),
                        thrust::device_ptr<uint64_t>(gaussianKeys.data() + gaussianKeys.size()),
                        thrust::device_ptr<uint32_t>(gaussianKeysIdx.data()));
    // --

    // GET TILES RANGES AMONG GAUSSIANS
    thrust::fill(thrust::cuda::par.on(rmm::cuda_stream_default),
                tileRanges.begin(),
                tileRanges.end(),
                TileRange{-1, -1});
    GS = (gaussianKeys.size() + BS - 1) / BS;
    IdentifyTileRanges<<<GS,BS>>>(gaussianKeys, tileRanges);
    // --

    // BLEND
    thrust::fill(thrust::cuda::par.on(rmm::cuda_stream_default.value()),
             d_image.begin(),
             d_image.end(),
             0.0f);
    dim3 BSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 GSize((camera.width+TILE_SIZE-1)/TILE_SIZE, (camera.height+TILE_SIZE-1)/TILE_SIZE, 1);
    AlphaBlend<<<GSize, BSize>>>(tileRanges, gaussiansSplateds, gaussianKeysIdx, camera.width, camera.height, d_image.data());
    cudaMemcpyAsync(image, d_image.data(), d_image.size() * sizeof(float), cudaMemcpyDeviceToHost, rmm::cuda_stream_default);
    // --
}