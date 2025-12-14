#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Kernel prototype
        __global__ void naiveScanStep(int *odata, const int *idata, int n, int offset);

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			int *dev_in, *dev_out;
            cudaMalloc(&dev_in, n * sizeof(int));
			cudaMalloc(&dev_out, n * sizeof(int));

			// copy input to device
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize =128;
            int numBlocks = (n + blockSize -1) / blockSize;

            // perform inclusive naive scan iteratively: dev_out = f(dev_in), then swap
            for (int offset =1; offset < n; offset *=2) {
                naiveScanStep<<<numBlocks, blockSize>>>(dev_out, dev_in, n, offset);
                cudaDeviceSynchronize();
                std::swap(dev_in, dev_out);
            }

			// Now dev_in holds the inclusive scan result. Convert to exclusive by shifting right into dev_out
			if (n >0) {
				// set first element to0
				cudaMemset(dev_out,0, sizeof(int));
				if (n >1) {
					cudaMemcpy(dev_out +1, dev_in, (n -1) * sizeof(int), cudaMemcpyDeviceToDevice);
				}
			}

			// copy exclusive scan back to host
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_in);
			cudaFree(dev_out);

            timer().endGpuTimer();
        }

        __global__ void naiveScanStep(int *odata, const int *idata, int n, int offset) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
		}
    }
}
