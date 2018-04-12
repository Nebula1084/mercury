#include <asian/MonteCarlo.h>
#include <asian/SumReduction.cuh>

#define BLOCK_N 256
#define THREAD_N 256

__global__ void monteCarloOptionKernel(Asian *asian, double *sum, double *sum2)
{
    __shared__ double sumThread[THREAD_N];
    __shared__ double sum2Thread[THREAD_N];
    double sumPerThread = 0;
    double sum2PerThread = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < asian->pathNum; i += blockDim.x * gridDim.x)
    {
        sumPerThread += 1;
        sum2PerThread += 1;
    }
    sumThread[threadIdx.x] = sumPerThread;
    sum2Thread[threadIdx.x] = sum2PerThread;
    sumReduce<double, THREAD_N, THREAD_N>(sumThread, sum2Thread);

    if (threadIdx.x == 0)
    {
        sum[blockIdx.x] = sumThread[0];
        sum2[blockIdx.x] = sum2Thread[0];        
    }
}

double monteCarloGPU(Asian *asian)
{
    Asian *option;
    double *sum;
    double *sumHost;
    double *sum2;
    double ret = 0;

    cudaMalloc(&option, sizeof(Asian));
    cudaMallocHost(&sum, sizeof(double) * BLOCK_N);
    cudaMallocHost(&sum2, sizeof(double) * BLOCK_N);

    cudaMemcpy(option, asian, sizeof(Asian), cudaMemcpyHostToDevice);

    monteCarloOptionKernel<<<BLOCK_N, THREAD_N>>>(option, sum, sum2);
    sumHost = new double[BLOCK_N];
    cudaMemcpy(sumHost, sum, BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < BLOCK_N; i++)
    {
        ret += sumHost[i];
    }

    
    delete[] sumHost;
    cudaFree(option);
    cudaFreeHost(sum);
    cudaFreeHost(sum2);
    
    return ret;
}
