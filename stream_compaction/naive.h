#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int* odata, const int* idata);

#if defined(__CUDACC__)
        __global__ void naiveScanStep(int* odata, const int* idata, int n, int offset);
#endif
    }


}