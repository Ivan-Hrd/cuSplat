#include "tiling.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "camera.hpp"
#include "gaussian.hpp"
#include "raft/core/device_span.hpp"

__global__ void IdentifyTileRanges(const raft::device_span<uint64_t> gaussianKeys, const raft::device_span<TileRange> tileRanges) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= gaussianKeys.size())
        return;
    uint32_t tile_id = (uint32_t)(gaussianKeys[i] >> 32);
    uint32_t prev_id = i > 0 ? (uint32_t)(gaussianKeys[i-1] >> 32) : tile_id;
    if (i == 0 || tile_id != prev_id)
        tileRanges[tile_id].start = i;
    if (i == (int)gaussianKeys.size()-1 || tile_id != (uint32_t)(gaussianKeys[i+1] >> 32))
        tileRanges[tile_id].end = i+1;

}

std::vector<TileRange> IdentifyTileRanges(const std::vector<GaussianKey>& gaussianKeys, Camera& camera) {
    int nTilesX = (camera.width+TILE_SIZE-1) / TILE_SIZE;
    int nTilesY = (camera.height+TILE_SIZE-1) / TILE_SIZE;
    std::vector<TileRange> tileRanges(nTilesX*nTilesY, {-1, -1});
    for (int i = 0; i < (int)gaussianKeys.size(); i++) {
        uint32_t tile_id = (uint32_t)(gaussianKeys[i].key >> 32);
        uint32_t prev_id = i > 0 ? (uint32_t)(gaussianKeys[i-1].key >> 32) : tile_id;
        if (i == 0 || tile_id != prev_id)
            tileRanges[tile_id].start = i;
        if (i == (int)gaussianKeys.size()-1 || tile_id != (uint32_t)(gaussianKeys[i+1].key >> 32))
            tileRanges[tile_id].end = i+1;
    }
    return tileRanges;
}
__global__ void GetSize(const raft::device_span<GaussianSplated> gaussian_splateds, int nTilesX, int nTilesY, int* counter) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= gaussian_splateds.size())
        return;
    int tile_min_x = (int)floorf((gaussian_splateds[i].u - gaussian_splateds[i].radius) / TILE_SIZE);
    int tile_max_x = (int)ceilf((gaussian_splateds[i].u + gaussian_splateds[i].radius) / TILE_SIZE) -1;
    int tile_min_y = (int)floorf((gaussian_splateds[i].v - gaussian_splateds[i].radius) / TILE_SIZE);
    int tile_max_y = (int)ceilf((gaussian_splateds[i].v + gaussian_splateds[i].radius) / TILE_SIZE) -1;
    tile_min_x = max(0, tile_min_x);
    tile_max_x = min(nTilesX-1, tile_max_x);
    tile_min_y = max(0, tile_min_y);
    tile_max_y = min(nTilesY-1, tile_max_y);
    int tmpSize = 0;
    for (int y=tile_min_y; y <= tile_max_y; y++) {
        for (int x=tile_min_x; x <= tile_max_x; x++) {
            tmpSize++;
        }
    }
    atomicAdd(counter, tmpSize);
}

__global__ void FillKeys(const raft::device_span<GaussianSplated> gaussian_splateds, int nTilesX, int nTilesY,
                         const raft::device_span<uint64_t> gaussianKeys, const raft::device_span<uint32_t> gaussianKeysIdx, int* count) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= gaussian_splateds.size())
        return;
    int tile_min_x = (int)floorf((gaussian_splateds[i].u - gaussian_splateds[i].radius) / TILE_SIZE);
    int tile_max_x = (int)ceilf((gaussian_splateds[i].u + gaussian_splateds[i].radius) / TILE_SIZE) -1;
    int tile_min_y = (int)floorf((gaussian_splateds[i].v - gaussian_splateds[i].radius) / TILE_SIZE);
    int tile_max_y = (int)ceilf((gaussian_splateds[i].v + gaussian_splateds[i].radius) / TILE_SIZE) -1;
    tile_min_x = max(0, tile_min_x);
    tile_max_x = min(nTilesX-1, tile_max_x);
    tile_min_y = max(0, tile_min_y);
    tile_max_y = min(nTilesY-1, tile_max_y);
    for (int y=tile_min_y; y <= tile_max_y; y++) {
        for (int x=tile_min_x; x <= tile_max_x; x++) {
            uint64_t tile_id = y * nTilesX + x;  // unique tile index
            uint32_t depth_bits;
            std::memcpy(&depth_bits, &gaussian_splateds[i].depth, sizeof(float)); // float into bits
            uint64_t key = (tile_id << 32) | depth_bits;
            int pos = atomicAdd(count, 1);
            gaussianKeys[pos] = key;
            gaussianKeysIdx[pos] = i;
        }
    }

}
std::vector<GaussianKey> CreateTiles(std::vector<GaussianSplated>& gaussian_splateds, Camera& camera) {
    std::vector<GaussianKey> gaussianKeys;
    int nTilesX = (camera.width+TILE_SIZE-1) / TILE_SIZE;
    int nTilesY = (camera.height+TILE_SIZE-1) / TILE_SIZE;
    for (u_int32_t i=0; i < gaussian_splateds.size(); i++) {
        int tile_min_x = (int)std::floor((gaussian_splateds[i].u - gaussian_splateds[i].radius) / TILE_SIZE);
        int tile_max_x = (int)std::ceil((gaussian_splateds[i].u + gaussian_splateds[i].radius) / TILE_SIZE) -1;
        int tile_min_y = (int)std::floor((gaussian_splateds[i].v - gaussian_splateds[i].radius) / TILE_SIZE);
        int tile_max_y = (int)std::ceil((gaussian_splateds[i].v + gaussian_splateds[i].radius) / TILE_SIZE) -1;
        tile_min_x = std::max(0, tile_min_x);
        tile_max_x = std::min(nTilesX-1, tile_max_x);
        tile_min_y = std::max(0, tile_min_y);
        tile_max_y = std::min(nTilesY-1, tile_max_y);
        for (int y=tile_min_y; y <= tile_max_y; y++) {
            for (int x=tile_min_x; x <= tile_max_x; x++) {
                uint64_t tile_id = y * nTilesX + x;  // unique tile index
                uint32_t depth_bits;
                std::memcpy(&depth_bits, &gaussian_splateds[i].depth, sizeof(float)); // float into bits
                uint64_t key = (tile_id << 32) | depth_bits;
                gaussianKeys.push_back(GaussianKey(key, i));
            }
        }
    }
    std::sort(gaussianKeys.begin(), gaussianKeys.end(), [](GaussianKey a, GaussianKey b) {return a.key<b.key;});
    return gaussianKeys;

}
