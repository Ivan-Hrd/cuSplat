// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#include "tiling.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "camera.hpp"
#include "gaussian.hpp"

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

std::vector<GaussianKey> CreateTiles(std::vector<GaussianSplated>& gaussian_splateds, Camera& camera) {
    std::vector<GaussianKey> gaussianKeys;
    int nTilesX = (camera.width+TILE_SIZE-1) / TILE_SIZE;
    int nTilesY = (camera.height+TILE_SIZE-1) / TILE_SIZE;
    for (int i=0; i < gaussian_splateds.size(); i++) {
        int tile_min_x = (int)std::floor(gaussian_splateds[i].u - gaussian_splateds[i].radius) / TILE_SIZE;
        int tile_max_x = (int)std::ceil(gaussian_splateds[i].u + gaussian_splateds[i].radius) / TILE_SIZE;
        int tile_min_y = (int)std::floor(gaussian_splateds[i].v - gaussian_splateds[i].radius) / TILE_SIZE;
        int tile_max_y = (int)std::ceil(gaussian_splateds[i].v + gaussian_splateds[i].radius) / TILE_SIZE;
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
