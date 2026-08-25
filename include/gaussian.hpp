#pragma once

#include <string>
#include <vector>

struct Gaussian {
    float x, y, z;
    float f_dc[3];
    float f_rest[45];
    float opacity;
    float sx, sy, sz;
    float rx, ry, rz, rw;
};

struct GaussianSplated { // 2D
    float u, v;
    float color[3];
    float opacity;
    float cov[2][2]; // rotation & scale
    float depth;
    float radius;
};

void loadPLY(const std::string& filename, Gaussian* gaussians);
size_t getSize(const std::string& filename);