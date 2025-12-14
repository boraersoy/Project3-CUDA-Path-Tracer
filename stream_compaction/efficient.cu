#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <algorithm>

namespace StreamCompaction {
	namespace Efficient {
		// Refer to timer type by fully qualified name to avoid lookup problems
		StreamCompaction::Common::PerformanceTimer& timer()
		{
			static StreamCompaction::Common::PerformanceTimer timer;
			return timer;
		}

		__global__ void upsweep(int n, int* data, int d, int totalItems) {
			int gid = blockIdx.x * blockDim.x + threadIdx.x;
			if (gid >= totalItems) return;

			int right = ((gid + 1) << (d + 1)) - 1;
			int left = right - (1 << d);

			if (right < n)
				data[right] += data[left];
		}


		__global__ void downsweep(int n, int* data, int d, int totalItems) {
			int gid = blockIdx.x * blockDim.x + threadIdx.x;
			if (gid >= totalItems) return;

			int right = ((gid + 1) << (d + 1)) - 1;
			int left = right - (1 << d);

			int t = data[left];
			data[left] = data[right];
			data[right] += t;
		}

		// simple kernel to invert0/1 integers on device
		__global__ void invertInts(int n, const int* in, int* out) {
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= n) return;
			out[i] = in[i] ? 0 : 1;
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int* odata, const int* idata, const bool record_time) {
			if (record_time) {
				timer().startGpuTimer();

			}

			int n_padded = 1 << ilog2ceil(n);
			int* dev_data;
			cudaMalloc((void**)&dev_data, n_padded * sizeof(int));
			// copy input from host to device
			if (n > 0) {
				cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			}

			if (n_padded > n)
				cudaMemset(dev_data + n, 0, (n_padded - n) * sizeof(int));


			int blockSize = 128;
			int maxD = ilog2ceil(n_padded);
			// Upsweep
			for (int d = 0; d < maxD; d++) {
				int totalItems = n_padded >> (d + 1);
				if (totalItems == 0) continue;

				int threadsPerBlock = std::min(blockSize, totalItems);
				int numBlocks = (totalItems + threadsPerBlock - 1) / threadsPerBlock;

				upsweep << <numBlocks, threadsPerBlock >> > (n_padded, dev_data, d, totalItems);
				cudaDeviceSynchronize();
			}

			// Set last element to0 for exclusive scan
			cudaMemset(dev_data + n_padded - 1, 0, sizeof(int));


			// Downsweep
			for (int d = maxD - 1; d >= 0; d--) {
				int totalItems = n_padded >> (d + 1);
				if (totalItems == 0) continue;
				int threadsPerBlock = std::min(blockSize, totalItems);

				int numBlocks = (totalItems + threadsPerBlock - 1) / threadsPerBlock;
				if (numBlocks > 0) {
					downsweep << <numBlocks, threadsPerBlock >> > (n_padded, dev_data, d, totalItems);
					cudaDeviceSynchronize();
				}
			}

			// Copy only first n elements back
			if (n > 0) {
				cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			}

			cudaFree(dev_data);
			if (record_time) {
				timer().endGpuTimer();
			}
		}


		int compact(int n, int* odata, const int* idata) {
			timer().startGpuTimer();

			int* dev_bool = nullptr; int* dev_scan = nullptr; int* dev_data = nullptr; int* out_data = nullptr;
			cudaMalloc((void**)&dev_bool, n * sizeof(int));
			cudaMalloc((void**)&dev_scan, n * sizeof(int));
			cudaMalloc((void**)&dev_data, n * sizeof(int));
			cudaMalloc((void**)&out_data, n * sizeof(int));

			// copy input from host to device
			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(dev_bool, 0, n * sizeof(int));
			cudaMemset(dev_scan, 0, n * sizeof(int));
			cudaMemset(out_data, 0, n * sizeof(int));

			int blockSize = 128;
			dim3 fullBlocks((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <fullBlocks, blockSize >> > (n, dev_bool, dev_data);
			scan(n, dev_scan, dev_bool, false);
			StreamCompaction::Common::kernScatter << <fullBlocks, blockSize >> > (n, out_data, dev_data, dev_bool, dev_scan);
			cudaMemcpy(odata, out_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			int lastBool = 0, lastScan = 0;
			if (n > 0) cudaMemcpy(&lastBool, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			if (n > 0) cudaMemcpy(&lastScan, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			int count = lastBool + lastScan;

			cudaFree(dev_bool);
			cudaFree(dev_scan);
			cudaFree(dev_data);
			cudaFree(out_data);
			timer().endGpuTimer();
			return count;
		}
	}
}
