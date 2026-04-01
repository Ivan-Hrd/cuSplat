#ifndef RT_CORE_GAUSSIAN_SPLATTING_GAUSSIAN_H
#define RT_CORE_GAUSSIAN_SPLATTING_GAUSSIAN_H
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

#endif //RT_CORE_GAUSSIAN_SPLATTING_GAUSSIAN_H
