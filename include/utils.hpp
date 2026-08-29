#pragma once

#include <vector>
#include <memory>
#include <span>

#include "camera.hpp"
#include "gaussian.hpp"
#include "tiling.hpp"
#include "raft/core/device_span.hpp"
#include "rmm/device_uvector.hpp"

__device__ inline Gaussian worldToCamera(Gaussian g, const Camera& cam) {
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

__device__ inline bool isVisible(Gaussian g, const Camera& cam, float radius) {
    if (g.z <= cam.near || g.z >= cam.far) {
        return false;
    }
    float u = cam.fx * (g.x/g.z) + cam.cx;
    float v = cam.fy * (g.y/g.z) + cam.cy;
    return u+radius > 0 && u-radius < cam.width &&
           v+radius > 0 && v-radius < cam.height;
}
__global__ void screenspaceGaussians(const raft::device_span<Gaussian> gaussians, const Camera camera, const raft::device_span<GaussianSplated> res);


