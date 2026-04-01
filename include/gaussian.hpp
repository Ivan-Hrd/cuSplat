// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.
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

std::vector<Gaussian> loadPLY(const std::string& filename);