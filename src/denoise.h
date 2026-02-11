#pragma once

#include <vector>
#include "scene.h"


void showGBuffer(uchar4* pbo);
void showNBuffer(uchar4* pbo);
void showPBuffer(uchar4* pbo);
void denoise(uchar4* image, int filterSize, int iter, float c_phi,
    float n_phi, float p_phi);

void showImage(uchar4* pbo, int iter); 

