#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <iostream>
#include <algorithm>

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata, const bool record_time) {
            if (record_time) {
                timer().startCpuTimer();

            }
            for(int i = 0; i < n; i++) {
                if(i == 0) {
                    odata[i] = 0;
                } else {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
			}
            if (record_time) {
                timer().endCpuTimer();

            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
             timer().startCpuTimer();
            
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* temp = new int[n];
            int* scanResult = new int[n];
            for (int i = 0; i < n; i++) {
                temp[i] = (idata[i] != 0) ? 1 : 0;
				scanResult[i] = 0;

            }
            
            //Scan
			// stop outer timer so scan can start its own CPU timer without throwing
            scan(n, scanResult, temp, false);
			// resume outer timer
			int count = scanResult[n - 1] + temp[n - 1];
            
			//Scatter
            for (int j = 0; j < n; j++) {
                if (temp[j] == 1) odata[scanResult[j]] = idata[j];
                
            }
            timer().endCpuTimer();
			delete[] temp;
            delete[] scanResult;
            return count;
        }

        void radixSort(int n, int* odata, const int* idata) {
            timer().startCpuTimer();

            int* in = new int[n];
            memcpy(in, idata, n * sizeof(int));

            int* out = new int[n];

            // Find max bits
            int maxVal = 0;
            for (int i = 0; i < n; i++) maxVal = std::max(maxVal, in[i]);
            int maxBits = 0;
            while ((1 << maxBits) <= maxVal) maxBits++;

            int* bools = new int[n];
            int* invbools = new int[n];
            int* falseArray = new int[n];

            for (int bit = 0; bit < maxBits; bit++) {

                // Compute 0/1 buckets
                for (int i = 0; i < n; i++) {
                    bools[i] = (in[i] >> bit) & 1;
                    invbools[i] = 1 - bools[i];   // 1 = zero-bit, 0 = one-bit
                }

                // Scan to find positions of zero bits
                scan(n, falseArray, invbools, false);

                int totalFalse = falseArray[n - 1] + invbools[n - 1];
                int trueCounter = 0;

                // Scatter
                for (int i = 0; i < n; i++) {
                    if (bools[i] == 0) {
                        // Zero: use scanned index
                        out[falseArray[i]] = in[i];
                    }
                    else {
                        // One: place after the zeros
                        int index = totalFalse + trueCounter;
                        out[index] = in[i];
                        trueCounter++;
                    }
                }

                // Prepare for next bit
                memcpy(in, out, n * sizeof(int));
            }

            memcpy(odata, in, n * sizeof(int));

            delete[] in;
            delete[] out;
            delete[] bools;
            delete[] invbools;
            delete[] falseArray;

            timer().endCpuTimer();
        }

    }
}
