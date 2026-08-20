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
    if (py_global >= height || px_global >= width) {
        return;
    }
    unsigned int t = blockIdx.x + gridDim.x * blockIdx.y;
    int start = tileRanges[t].start;
    int end = tileRanges[t].end;
    if (start == -1) {
        return;
    }
    float color[3] = {0.0f, 0.0f, 0.0f};
    float T = 1;
    for (unsigned int gaussian=start; gaussian<end; gaussian++) {
        GaussianSplated gaussianSplated = gaussSplateds[gaussianKeysIdx[gaussian]];
        float alpha = computeAlpha(gaussianSplated, px_global, py_global);
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
        color[0] += gaussianSplated.color[0]*T*alpha;
        color[1] += gaussianSplated.color[1]*T*alpha;
        color[2] += gaussianSplated.color[2]*T*alpha;
        T *= (1-alpha);
        if (T < 0.0001f) {
            break;
        }
    }
    for (int c = 0; c < 3; c++) {
        image[(py_global*width+px_global)*3+c] = color[c];
    }
}

void rasterize(const std::vector<Gaussian>& gaussians, Camera &camera, float *image) {
    // CULLING
    auto cullGaussiansMemory = rmm::device_uvector<Gaussian>(gaussians.size(), rmm::cuda_stream_default);
    thrust::device_vector<Gaussian> d_gaussians = gaussians;
    auto end = thrust::copy_if(d_gaussians.begin(), d_gaussians.end(),
                                    cullGaussiansMemory.begin(), [camera] __device__(Gaussian gaussian) {
        gaussian = worldToCamera(gaussian, camera);
        float radius = camera.fx * max(max(gaussian.sx, gaussian.sy),gaussian.sz) / gaussian.z;
        return isVisible(gaussian, camera, radius);
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
    int nTilesX = (camera.width+TILE_SIZE-1) / TILE_SIZE;
    int nTilesY = (camera.height+TILE_SIZE-1) / TILE_SIZE;
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
    auto tileRangesMemory = rmm::device_uvector<TileRange>(nTilesX*nTilesY, rmm::cuda_stream_default);
    auto tileRanges = raft::device_span<TileRange>(tileRangesMemory.data(), nTilesX*nTilesY);
    thrust::fill(thrust::cuda::par.on(rmm::cuda_stream_default),
                tileRangesMemory.begin(),
                tileRangesMemory.end(),
                TileRange{-1, -1});
    GS = (gaussianKeys.size() + BS - 1) / BS;
    IdentifyTileRanges<<<GS,BS>>>(gaussianKeys, tileRanges);
    // --

    // BLEND
    auto d_image = rmm::device_uvector<float>(camera.width*camera.height*3, rmm::cuda_stream_default);
    thrust::fill(thrust::cuda::par.on(rmm::cuda_stream_default.value()),
             d_image.begin(),
             d_image.end(),
             0.0f);
    dim3 BSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 GSize((camera.width+TILE_SIZE-1)/TILE_SIZE, (camera.height+TILE_SIZE-1)/TILE_SIZE, 1);
    AlphaBlend<<<GSize, BSize>>>(tileRanges, gaussiansSplateds, gaussianKeysIdx, camera.width, camera.height, d_image.data());
    cudaMemcpy(image, d_image.data(), d_image.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // --
}