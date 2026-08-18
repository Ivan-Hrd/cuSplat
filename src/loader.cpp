#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "gaussian.hpp"

#define NB_ATTRIBUTES 62

std::vector<Gaussian> loadPLY(const std::string& filename) {
    /*
     *  Read from a PLY file that must follow : format binary_little_endian 1.0,
     *  62 floats for each gaussian : 3 coordinates + 3 normals + 3 f_dc + 45 SH + 1 opacity + 3 scale + 4 rotation (quaternions)
     *  Returns: A vector of gaussian struct contained in the file
     */
    std::vector<Gaussian> result;
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + filename);

    size_t nbGauss = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos)
            nbGauss = std::stoi(line.substr(15));
        if (line == "end_header") break;
    }

    size_t off = 0;
    std::vector<char> allData(sizeof(float) * NB_ATTRIBUTES * nbGauss);
    file.read(allData.data(),allData.size());
    if (!file) throw std::runtime_error("Error reading binary data");
    for (unsigned int i = 0; i < nbGauss; ++i) {
        Gaussian gaussian;
        // get coordinates
        std::memcpy(&gaussian.x, allData.data() + off, sizeof(float));
        off += sizeof(float);
        std::memcpy(&gaussian.y, allData.data() + off, sizeof(float));
        off += sizeof(float);
        std::memcpy(&gaussian.z, allData.data() + off, sizeof(float));
        off += sizeof(float);

        // skip normals
        off += sizeof(float)*3;

        // get base color
        std::memcpy(&gaussian.f_dc[0], allData.data() + off, sizeof(float));
        off += sizeof(float);
        std::memcpy(&gaussian.f_dc[1], allData.data() + off, sizeof(float));
        off += sizeof(float);
        std::memcpy(&gaussian.f_dc[2], allData.data() + off, sizeof(float));
        off += sizeof(float);

        // get spherical harmonics
        unsigned int nbSphericalHarmonics = 45;
        for (unsigned int sh = 0; sh < nbSphericalHarmonics; ++sh) {
            std::memcpy(&gaussian.f_rest[sh], allData.data() + off, sizeof(float));
            off += sizeof(float);
        }

        // get opacity
        std::memcpy(&gaussian.opacity, allData.data() + off, sizeof(float));
        gaussian.opacity = 1.0f / (1.0f + std::exp(-gaussian.opacity)); // apply sigmoid
        off += sizeof(float);

        // get scale (stored in log)
        std::memcpy(&gaussian.sx, allData.data() + off, sizeof(float));
        gaussian.sx = std::exp(gaussian.sx);
        off += sizeof(float);
        std::memcpy(&gaussian.sy, allData.data() + off, sizeof(float));
        gaussian.sy = std::exp(gaussian.sy);
        off += sizeof(float);
        std::memcpy(&gaussian.sz, allData.data() + off, sizeof(float));
        gaussian.sz = std::exp(gaussian.sz);
        off += sizeof(float);

        // get rotation (quaternions)
        std::memcpy(&gaussian.rw, allData.data() + off, sizeof(float));
        off += sizeof(float);
        std::memcpy(&gaussian.rx, allData.data() + off, sizeof(float));
        off += sizeof(float);
        std::memcpy(&gaussian.ry, allData.data() + off, sizeof(float));
        off += sizeof(float);
        std::memcpy(&gaussian.rz, allData.data() + off, sizeof(float));
        off += sizeof(float);
        // normalize quaternions
        float norm = std::sqrt(gaussian.rx*gaussian.rx + gaussian.ry*gaussian.ry
                             + gaussian.rz*gaussian.rz + gaussian.rw*gaussian.rw);
        gaussian.rx /= norm;
        gaussian.ry /= norm;
        gaussian.rz /= norm;
        gaussian.rw /= norm;

        result.push_back(gaussian);
    }

    assert(result.size() == nbGauss);
    return result;
}
