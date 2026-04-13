// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#pragma once
#include <cstdint>
#include <vector>

#include "camera.hpp"
#include "gaussian.hpp"

#define TILE_SIZE 16

struct GaussianKey {
    uint64_t key; // tileId | depth
    uint32_t index; // index in the original list
};

struct TileRange {
    int start;
    int end;
};

std::vector<TileRange> IdentifyTileRanges(const std::vector<GaussianKey>& gaussianKeys, Camera& camera);

std::vector<GaussianKey> CreateTiles(std::vector<GaussianSplated>& gaussian_splateds, Camera& camera);
