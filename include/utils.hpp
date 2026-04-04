// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.
#pragma once

#include <vector>

#include "camera.hpp"
#include "gaussian.hpp"

Gaussian worldToCamera(Gaussian g, Camera& cam);
std::vector<Gaussian> cullGaussian(std::vector<Gaussian>& gaussians, Camera camera);
std::vector<GaussianSplated> screenspaceGaussians(std::vector<Gaussian>& gaussians, Camera& camera);
void rasterize(std::vector<Gaussian> gaussians, Camera &camera, float *image);