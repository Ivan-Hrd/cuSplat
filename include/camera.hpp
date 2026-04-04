// © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

#pragma once

struct Camera {
    float pos[3];
    float R[3][3]; // rotation matrix world->camera

    int   width, height;
    float fx, fy; // focales in pixels (determine zoom)
    float cx, cy; // image center

    float near = 0.2f;
    float far  = 100.0f;
    Camera(float pos[3], float R[3][3], int width, int height,
           float fx, float fy, float cx, float cy,
           float near = 0.2f, float far = 100.0f)
        : width(width), height(height),
          fx(fx), fy(fy), cx(cx), cy(cy),
          near(near), far(far)
    {
        for (int i = 0; i < 3; i++)
            this->pos[i] = pos[i];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                this->R[i][j] = R[i][j];
    }

};