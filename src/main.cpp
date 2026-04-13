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
    width = 1959;
    height = 1090;
    float pos[3] = {-2.5199776022057296f, -0.09704735754873686f, -3.6247725540304545f};
    float R[3][3] = {
        { 0.9982731285632193f,  -0.011928707708098955f, -0.05751927260507243f},
        { 0.0065061360949636f,   0.9955928229282383f,   -0.09355533724430458f},
        { 0.0583817692581828f,   0.09301955098900708f,   0.9939511719154457f }
    };
    Camera cam(pos, R, width, height, 1159.5880733038064f, 1164.6601287484507f, 979.5f, 545.0f);
    float *image = new float[width * height * 3]();
    rasterize(gaussians, cam, image);
    savePPM("/home/h/CLionProjects/RT-Core-Gaussian-Splatting/output.ppm", image, width, height);
    delete[] image;
    return 0;
}
