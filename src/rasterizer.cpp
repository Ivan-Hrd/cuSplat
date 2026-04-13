// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <ostream>

#include "camera.hpp"
#include "utils.hpp"
#include "gaussian.hpp"
#include "tiling.hpp"

float computeAlpha(const GaussianSplated& g, float px, float py) {
    float alpha = 0.0f;
    float dx = px-g.u;
    float dy = py-g.v;
    // sigmaPrime inversion
    float det = g.cov[0][0]*g.cov[1][1] - g.cov[0][1]*g.cov[1][0];
    if (std::abs(det) < 1e-6f) return 0.0f;
    float power = (1/det) * (g.cov[1][1]*dx*dx - 2*g.cov[0][1]*dx*dy + g.cov[0][0]*dy*dy);
    if (power < 0.0f) return 0.0f;
    return g.opacity*std::exp(-0.5f*power);
}

void rasterize(const std::vector<Gaussian>& gaussians, Camera &camera, float *image) {
    auto cullGaussians = cullGaussian(gaussians, camera);
    std::vector<GaussianSplated> gaussSplateds = screenspaceGaussians(cullGaussians, camera);
    std::vector<GaussianKey> gaussianKeys = CreateTiles(gaussSplateds, camera);
    std::vector<TileRange> tileRanges = IdentifyTileRanges(gaussianKeys, camera);
    int nbTiles = tileRanges.size();
    for (int t = 0; t < nbTiles; t++) {
        int nTilesX = (camera.width + TILE_SIZE-1) / TILE_SIZE;
        int tile_x = t % nTilesX;
        int tile_y = t / nTilesX;
        for (int px = 0; px < TILE_SIZE; px++) {
            int px_global = tile_x * TILE_SIZE + px;
            if (px_global >= camera.width) {
                break;
            }
            for (int py = 0; py < TILE_SIZE; py++) {
                // global px & py
                int py_global = tile_y * TILE_SIZE + py;
                if (py_global >= camera.height) {
                    break;
                }
                int start = tileRanges[t].start;
                int end = tileRanges[t].end;
                if (start == -1) {
                    continue;
                }
                std::vector<float> color(3, 0);
                float T = 1;
                for (unsigned int gaussian=start; gaussian<end; gaussian++) {
                    GaussianSplated gaussianSplated = gaussSplateds[gaussianKeys[gaussian].index];
                    float alpha = computeAlpha(gaussianSplated, px_global, py_global);
                    alpha = std::clamp(alpha, 0.0f, 1.0f);
                    color[0] += gaussianSplated.color[0]*T*alpha;
                    color[1] += gaussianSplated.color[1]*T*alpha;
                    color[2] += gaussianSplated.color[2]*T*alpha;
                    T *= (1-alpha);
                    if (T < 0.0001f) {
                        break;
                    }
                }
                for (int c = 0; c < 3; c++) {
                    image[(py_global*camera.width+px_global)*3+c] = color[c];
                }
            }
        }
    }


}