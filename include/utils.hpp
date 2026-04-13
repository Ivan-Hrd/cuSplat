// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.
#pragma once

#include <vector>

#include "camera.hpp"
#include "gaussian.hpp"

Gaussian worldToCamera(Gaussian g, Camera& cam);
std::vector<Gaussian> cullGaussian(const std::vector<Gaussian>& gaussians, Camera camera);
std::vector<GaussianSplated> screenspaceGaussians(const std::vector<Gaussian>& gaussians, Camera& camera);
void rasterize(const std::vector<Gaussian>& gaussians, Camera &camera, float *image);