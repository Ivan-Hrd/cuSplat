// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#include <fstream>
#include <iostream>

#include "camera.hpp"
#include "gaussian.hpp"
#include "utils.hpp"

void savePPM(const std::string& filename, float* image, int width, int height) {
    std::ofstream f(filename);
    f << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; i++)
        f << (int)(std::min(1.0f, image[i]) * 255) << " ";
}

int main()
{
    std::cout << "Loading PLY file" << std::endl;
    std::vector<Gaussian> gaussians = loadPLY("/home/h/CLionProjects/RT-Core-Gaussian-Splatting/point_cloud.ply");
    int width, height;
    width = 1920;
    height = 1080;
    float pos[3] = {0, 0, 0};
    float R[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    Camera cam(pos, R, width, height, 1000.0f, 1000.0f, 960.0f, 540.0f);

    float *image = new float[width * height * 3]();
    rasterize(gaussians, cam, image);
    savePPM("/home/h/CLionProjects/RT-Core-Gaussian-Splatting/output.ppm", image, width, height);
    delete[] image;
    return 0;
}
