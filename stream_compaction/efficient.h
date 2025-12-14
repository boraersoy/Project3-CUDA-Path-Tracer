#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();
#if defined(__CUDACC__)
		__global__ void upsweep(int n, int* data, int d, int totalItems);
		__global__ void downsweep(int n, int* data, int d, int totalItems);
        __global__ void invertInts(int n, const int* in, int* out);
#endif
        void scan(int n, int *odata, const int *idata, const bool record_time);

        int compact(int n, int *odata, const int *idata);

        void radixStep(int n, int bit, int* dev_out, int* dev_in);

        void radixSort(int n, int* dev_data);


    }
}
