#pragma once
#include <cstdint>
#include <memory>

#include "camera.hpp"
#include "gaussian.hpp"
#include "raft/core/device_span.hpp"

#define TILE_SIZE 16

struct GaussianKey {
    uint64_t key; // tileId | depth
    uint32_t index; // index in the original list
};

struct TileRange {
    int start;
    int end;
};

__global__ void IdentifyTileRanges(const raft::device_span<uint64_t> gaussianKeys, const raft::device_span<TileRange> tileRanges);
__global__ void FillKeys(const raft::device_span<GaussianSplated> gaussian_splateds, int nTilesX, int nTilesY,
                         const raft::device_span<uint64_t> gaussianKeys, const raft::device_span<uint32_t> gaussianKeysIdx, int* count);
__global__ void GetSize(const raft::device_span<GaussianSplated> gaussian_splateds, int nTilesX, int nTilesY, int* counter);