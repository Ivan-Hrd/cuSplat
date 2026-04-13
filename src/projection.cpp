// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#include <cmath>
#include <vector>

#include "camera.hpp"
#include "gaussian.hpp"
#include "utils.hpp"

void quatToMatrix(const float w, const float x, const float y, const float z, float R[3][3]) {
    R[0][0]=1-2*(y*y+z*z); R[0][1]=2*(x*y-w*z); R[0][2]=2*(x*z+w*y);
    R[1][0]=2*(x*y+w*z); R[1][1]=1-2*(x*x+z*z); R[1][2]=2*(y*z-w*x);
    R[2][0]=2*(x*z-w*y); R[2][1]=2*(y*z+w*x); R[2][2]=1-2*(x*x+y*y);
}

struct COV3D {
    float cov[3][3];
};

COV3D buildCov3D(const Gaussian& g) {
    float R[3][3];
    quatToMatrix(g.rw, g.rx, g.ry, g.rz, R);

    float sx = g.sx, sy = g.sy, sz = g.sz;

    // M = R · S
    float M[3][3];
    for (int i = 0; i < 3; i++) {
        M[i][0] = R[i][0] * sx;
        M[i][1] = R[i][1] * sy;
        M[i][2] = R[i][2] * sz;
    }

    // Σ = M · M.T
    COV3D cov;
    cov.cov[0][0] = M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2];
    cov.cov[0][1] = M[0][0]*M[1][0] + M[0][1]*M[1][1] + M[0][2]*M[1][2];
    cov.cov[0][2] = M[0][0]*M[2][0] + M[0][1]*M[2][1] + M[0][2]*M[2][2];
    cov.cov[1][1] = M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2];
    cov.cov[1][2] = M[1][0]*M[2][0] + M[1][1]*M[2][1] + M[1][2]*M[2][2];
    cov.cov[2][2] = M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2];
    cov.cov[2][1] = cov.cov[1][2];
    cov.cov[2][0] = cov.cov[0][2];
    cov.cov[1][0] = cov.cov[0][1];
    return cov;
}

void buildJ(float X, float Y, float Z, float fx, float fy, float J[2][3]) {
    J[0][0] = fx/Z; J[0][1] = 0; J[0][2] = -fx*X/(Z*Z);
    J[1][0] = 0; J[1][1] = fy/Z; J[1][2] = -fy*Y/(Z*Z);
}

std::vector<GaussianSplated> screenspaceGaussians(const std::vector<Gaussian>& gaussians, Camera& camera) {
    std::vector<GaussianSplated> result;
    for (unsigned int i = 0; i < gaussians.size(); i++) {
        GaussianSplated gaussian;
        Gaussian Gcam = worldToCamera(gaussians[i], camera);
        float u = camera.fx * (Gcam.x/Gcam.z) + camera.cx;
        float v = camera.fy * (Gcam.y/Gcam.z) + camera.cy;
        gaussian.u = u;
        gaussian.v = v;
        gaussian.depth = Gcam.z;
        gaussian.opacity = Gcam.opacity;
        COV3D cov = buildCov3D(gaussians[i]);
        float (*W)[3] = camera.R; // world->camera
        float J[2][3]; // camera -> screen
        buildJ(Gcam.x, Gcam.y, Gcam.z, camera.fx, camera.fy, J);

        // Σ' = J · W · Σ · W.T · J.T
        float T[2][3];
        // T = J · W
        for (int i1 = 0; i1 < 2; i1++)
            for (int j1 = 0; j1 < 3; j1++)
                T[i1][j1] = J[i1][0]*W[0][j1]+J[i1][1]*W[1][j1]+J[i1][2]*W[2][j1];

        float A[2][3];
        // A = T · Σ
        for (int i1 = 0; i1 < 2; i1++)
            for (int j1 = 0; j1 < 3; j1++)
                A[i1][j1] = T[i1][0]*cov.cov[0][j1]+T[i1][1]*cov.cov[1][j1]+T[i1][2]*cov.cov[2][j1];

        // Σ' = A . T.T
        for (int i1 = 0; i1 < 2; i1++)
            for (int j1 = 0; j1 < 2; j1++)
                gaussian.cov[i1][j1] = A[i1][0]*T[j1][0]+A[i1][1]*T[j1][1]+A[i1][2]*T[j1][2];

        float a = gaussian.cov[0][0];
        float b1 = gaussian.cov[0][1];
        float c = gaussian.cov[1][1];

        float mid = 0.5f * (a + c);
        float diff = 0.5f * (a - c);
        float lambda_max = mid + std::sqrt(diff*diff + b1*b1); // plus grande valeur propre

        gaussian.radius = std::ceil(3.0f * std::sqrt(lambda_max));
        // ensure sigmaPrime is definite positive so that opacity determinant
        // does not create artifacts
        gaussian.cov[0][0] += 0.3f;
        gaussian.cov[1][1] += 0.3f;

        float dx = camera.pos[0] - gaussians[i].x;
        float dy = camera.pos[1] - gaussians[i].y;
        float dz = camera.pos[2] - gaussians[i].z;
        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
        dx /= len; dy /= len; dz /= len; // normalize
        // base color
        static constexpr float SH_C0 = 0.28209479177387814f;
        static constexpr float SH_C1 = 0.4886025119029199f;
        static constexpr float SH_C2[5] = {1.0925484f, 1.0925484f, 0.3153916f, 1.0925484f, 0.5462742f};
        static constexpr float SH_C3[7] = {
            0.5900436f, 2.8906114f, 0.4570458f,
            0.3731763f, 0.4570458f, 1.4453057f, 0.5900436f
        };
        float r = SH_C0 * gaussians[i].f_dc[0];
        float g = SH_C0 * gaussians[i].f_dc[1];
        float b = SH_C0 * gaussians[i].f_dc[2];
        gaussian.color[0] = r;
        gaussian.color[1] = g;
        gaussian.color[2] = b;

        // 1st order
        for (int c = 0; c < 3; c++) { // 3 channel
            gaussian.color[c] += SH_C1 * gaussians[i].f_rest[0*3+c] * dy +
                                 SH_C1 * gaussians[i].f_rest[1*3+c] * dz +
                                 SH_C1 * gaussians[i].f_rest[2*3+c] * dx;
        }
        // 2nd order
        for (int c = 0; c < 3; c++) { // 3 channel
            gaussian.color[c] += SH_C2[0] * gaussians[i].f_rest[3*3+c] * dy*dx +
                                 SH_C2[1] * gaussians[i].f_rest[4*3+c] * dz*dy +
                                 SH_C2[2] * gaussians[i].f_rest[5*3+c] * (2*dz*dz - dx*dx - dy*dy) +
                                 SH_C2[3] * gaussians[i].f_rest[6*3+c] * dx*dz +
                                 SH_C2[4] * gaussians[i].f_rest[7*3+c] * (dx*dx - dy*dy);
        }
        // 3rd order
        for (int c = 0; c < 3; c++) { // 3 channel
            gaussian.color[c] += SH_C3[0] * gaussians[i].f_rest[8*3+c] * dy*(3*dx*dx-dy*dy) +
                                 SH_C3[1] * gaussians[i].f_rest[9*3+c] * dz*dy*dx +
                                 SH_C3[2] * gaussians[i].f_rest[10*3+c] * dy*(4*dz*dz - dx*dx - dy*dy) +
                                 SH_C3[3] * gaussians[i].f_rest[11*3+c] * dz*(2*dz*dz - 3*dx*dx - 3*dy*dy) +
                                 SH_C3[4] * gaussians[i].f_rest[12*3+c] * dx*(4*dz*dz - dy*dy - dx*dx)+
                                 SH_C3[5] * gaussians[i].f_rest[13*3+c] * dz*(dx*dx - dy*dy)+
                                 SH_C3[6] * gaussians[i].f_rest[14*3+c] * dx*(dx*dx - 3*dy*dy);
        }

        for (int c = 0; c < 3; c++) {
            gaussian.color[c] = std::max(0.0f, std::min(1.0f, gaussian.color[c] + 0.5f));
        }
        //gaussian.radius = std::ceil(3.0f * std::sqrt(std::max(gaussian.cov[0][0], gaussian.cov[1][1])));
        result.push_back(gaussian);
    }
    return result;
}