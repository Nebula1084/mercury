#include <american/American.h>

#include <stdio.h>

#define BLOCK_N 256
#define THREAD_N 256

__global__ void expiryValueKernel(American *plan, double *value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = plan->step;
    int strike = plan->instrument.strike;

    for (int i = idx; i <= step; i += blockDim.x * gridDim.x)
    {
        double d = plan->asset.price * exp(plan->vDt * (2.0f * i - step));
        if (plan->instrument.type == CALL)
            d = d - strike;
        else if (plan->instrument.type == PUT)
            d = strike - d;
        value[i] = (d > 0) ? d : 0;
    }
}

__global__ void binomialKernel(American *plan, int iter, double *value, double *next)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < iter; i += blockDim.x * gridDim.x)
    {
        next[i] = plan->puByDf * value[i + 1] + plan->pdByDf * value[i];
    }
}

double American::binomialGPU()
{
    American *plan;
    double *value, *next;
    double result;

    cudaMalloc(&plan, sizeof(American));
    cudaMalloc(&value, sizeof(double) * (step + 1));
    cudaMalloc(&next, sizeof(double) * (step + 1));

    cudaMemcpy(plan, this, sizeof(American), cudaMemcpyHostToDevice);

    expiryValueKernel<<<BLOCK_N, THREAD_N>>>(plan, value);
    for (int i = step; i > 0; i--)
    {
        double *tmp;
        binomialKernel<<<BLOCK_N, THREAD_N>>>(plan, i, value, next);
        tmp = value;
        value = next;
        next = tmp;
    }

    cudaMemcpy(&result, value, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(plan);
    cudaFree(value);
    cudaFree(next);

    return result;
}
