#include <asian/MonteCarlo.h>
#include <asian/SumReduction.cuh>

#define BLOCK_N 256
#define THREAD_N 256

__device__ void randNormal(
    Asian *asian,
    curandState *state,
    double *choMatrix,
    double *depend,
    double *independ)
{
    int size = asian->basketSize;

    for (int i = 0; i < size; i++)
    {
        independ[i] = curand_normal(state);
    }

    for (int i = 0; i < size; i++)
    {
        double corNormal = 0;
        for (int j = 0; j < size; j++)
        {
            corNormal += independ[j] * choMatrix[i * size + j];
        }
        depend[i] = corNormal;
    }
}

__global__ void monteCarloOptionKernel(
    Asian *asian,
    double *choMatrix,
    double *depend,
    double *independ,
    double *sum,
    double *sumOutput,
    double *sum2,
    double *sum2Output,
    double *sumX,
    double *sumXOutput)
{
    __shared__ double sumThread[THREAD_N];
    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = asian->basketSize;
    int offset = idx * size;

    curand_init(1230, idx, 0, &state);
    for (int i = 0; i < size; i++)
    {
        sum[offset + i] = 0;
        sum2[offset + i] = 0;
        sumX[offset + i] = 0;
    }

    for (int i = idx; i < asian->pathNum; i += blockDim.x * gridDim.x)
    {
        randNormal(asian, &state, choMatrix, depend + offset, independ + offset);

        for (int j = 0; j < size; j++)
        {
            double var = depend[offset + j];
            sum[offset + j] += var;
            sum2[offset + j] += var * var;
            for (int k = 0; k < size; k++)
            {
                double val = depend[offset + k];
                sumX[offset * size + j * size + k] += var * val;
            }
        }
    }

    for (int i = 0; i < size; i++)
    {
        sumThread[threadIdx.x] = sum[offset + i];
        sumReduce<double, THREAD_N, THREAD_N>(sumThread);
        if (threadIdx.x == 0)
        {
            sumOutput[blockIdx.x * size + i] = sumThread[0];
        }
        sumThread[threadIdx.x] = sum2[offset + i];
        sumReduce<double, THREAD_N, THREAD_N>(sumThread);
        if (threadIdx.x == 0)
        {
            sum2Output[blockIdx.x * size + i] = sumThread[0];
        }
        for (int j = 0; j < size; j++)
        {
            sumThread[threadIdx.x] = sumX[offset * size + i * size + j];
            sumReduce<double, THREAD_N, THREAD_N>(sumThread);
            if (threadIdx.x == 0)
            {
                sumXOutput[blockIdx.x * size * size + i * size + j] = sumThread[0];
            }
        }
    }
}

double monteCarloGPU(Asian *asian, double *expectation, double *covMatrix)
{
    Asian *option;
    double *choMatrix;

    double *depend;
    double *independ;
    double *sum;
    double *sumOutput;
    double *sumHost;
    double *sum2;
    double *sum2Output;
    double *sum2Host;
    double *sumX;
    double *sumXOutput;
    double *sumXHost;

    int size = asian->basketSize;

    cudaMalloc(&option, sizeof(Asian));
    cudaMalloc(&choMatrix, size * size * sizeof(double));

    int totalThread = BLOCK_N * THREAD_N;
    cudaMalloc(&depend, sizeof(double) * size * totalThread);
    cudaMalloc(&independ, sizeof(double) * size * totalThread);
    cudaMalloc(&sum, sizeof(double) * size * totalThread);
    cudaMalloc(&sum2, sizeof(double) * size * totalThread);
    cudaMalloc(&sumX, sizeof(double) * size * size * totalThread);

    cudaMalloc(&sumOutput, sizeof(double) * size * BLOCK_N);
    cudaMalloc(&sum2Output, sizeof(double) * size * BLOCK_N);
    cudaMalloc(&sumXOutput, sizeof(double) * size * size * BLOCK_N);

    cudaMemcpy(option, asian, sizeof(Asian), cudaMemcpyHostToDevice);
    cudaMemcpy(choMatrix, asian->choMatrix, size * size * sizeof(double), cudaMemcpyHostToDevice);

    monteCarloOptionKernel<<<BLOCK_N, THREAD_N>>>(
        option, choMatrix,
        depend, independ,
        sum, sumOutput,
        sum2, sum2Output,
        sumX, sumXOutput);

    cudaMallocHost(&sumHost, sizeof(double) * size * BLOCK_N);
    cudaMallocHost(&sum2Host, sizeof(double) * size * BLOCK_N);
    cudaMallocHost(&sumXHost, sizeof(double) * size * size * BLOCK_N);
    cudaMemcpy(sumHost, sumOutput, size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum2Host, sum2Output, size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sumXHost, sumXOutput, size * size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        expectation[i] = 0;
        for (int j = 0; j < size; j++)
        {
            covMatrix[i * size + j] = 0;
        }
    }
    for (int i = 0; i < BLOCK_N; i++)
    {
        for (int j = 0; j < size; j++)
        {
            expectation[j] += sumHost[i * size + j];
            for (int k = 0; k < size; k++)
            {
                covMatrix[j * size + k] += sumXHost[i * size * size + j * size + k];
            }
        }
    }

    for (int i = 0; i < size; i++)
    {
        expectation[i] /= asian->pathNum;
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            covMatrix[i * size + j] = covMatrix[i * size + j] / asian->pathNum - expectation[i] * expectation[j];
        }
    }

    cudaFreeHost(sumHost);
    cudaFreeHost(sum2Host);

    cudaFree(choMatrix);
    cudaFree(asian);
    cudaFree(depend);
    cudaFree(independ);
    cudaFree(sum);
    cudaFree(sum2);
    cudaFree(sumOutput);
    cudaFree(sum2Output);
    cudaFree(sumXOutput);
    return 0;
}
