// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#include <vector>

#include "camera.hpp"
#include "gaussian.hpp"

Gaussian worldToCamera(Gaussian g, Camera& cam) {
    Gaussian cameraGaussian = g;
    cameraGaussian.x -= cam.pos[0];
    cameraGaussian.y -= cam.pos[1];
    cameraGaussian.z -= cam.pos[2];
    float x = cameraGaussian.x*cam.R[0][0] + cameraGaussian.y*cam.R[0][1] + cameraGaussian.z*cam.R[0][2];
    float y = cameraGaussian.x*cam.R[1][0] + cameraGaussian.y*cam.R[1][1] + cameraGaussian.z*cam.R[1][2];
    float z = cameraGaussian.x*cam.R[2][0] + cameraGaussian.y*cam.R[2][1] + cameraGaussian.z*cam.R[2][2];
    cameraGaussian.x = x;
    cameraGaussian.y = y;
    cameraGaussian.z = z;
    return cameraGaussian;
}

bool isVisible(Gaussian g, Camera& cam, float radius) {
    if (g.z <= cam.near || g.z >= cam.far) {
        return false;
    }
    float u = cam.fx * (g.x/g.z) + cam.cx;
    float v = cam.fy * (g.y/g.z) + cam.cy;
    return u+radius > 0 && u-radius < cam.width &&
           v+radius > 0 && v-radius < cam.height;
}

std::vector<Gaussian> cullGaussian(const std::vector<Gaussian>& gaussians, Camera camera) {
    std::vector<Gaussian> result;
    for (unsigned int i = 0; i < gaussians.size(); i++) {
        Gaussian gaussian = gaussians[i];
        gaussian = worldToCamera(gaussian, camera);
        float radius = camera.fx * std::max(std::max(gaussian.sx, gaussian.sy),gaussian.sz) / gaussian.z;
        if (isVisible(gaussian, camera, radius)) {
            result.push_back(gaussians[i]);
        }
    }
    return result;
}