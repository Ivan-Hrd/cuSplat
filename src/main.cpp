// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#include <iostream>

#include "gaussian.hpp"

int main()
{
    std::cout << "Loading PLY file" << std::endl;
    std::vector<Gaussian> gaussians = loadPLY("/home/h/CLionProjects/RT-Core-Gaussian-Splatting/point_cloud.ply");
    for (size_t i = 0; i < 5; ++i) {
        const Gaussian& g = gaussians[i];

        std::cout << "\nGaussian #" << i << ":\n";

        std::cout << "  Position: (" << g.x << ", " << g.y << ", " << g.z << ")\n";

        std::cout << "  f_dc: ";
        for (int j = 0; j < 3; ++j)
            std::cout << g.f_dc[j] << " ";
        std::cout << "\n";

        std::cout << "  f_rest: ";
        for (int j = 0; j < 45; ++j)
            std::cout << g.f_rest[j] << " ";
        std::cout << "\n";

        std::cout << "  Opacity: " << g.opacity << "\n";

        std::cout << "  Scale: (" << g.sx << ", " << g.sy << ", " << g.sz << ")\n";

        std::cout << "  Rotation (quat): ("
                  << g.rx << ", " << g.ry << ", "
                  << g.rz << ", " << g.rw << ")\n";
    }
    return 0;
}
