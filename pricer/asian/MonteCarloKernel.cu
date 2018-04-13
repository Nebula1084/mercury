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
    double *sum2Output)
{
    __shared__ double sumThread[THREAD_N];
    double sumPerThread = 0;
    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = asian->basketSize;
    int offset = idx * size;

    curand_init(1234, idx, 0, &state);
    for (int i = 0; i < size; i++)
    {
        sum[offset + i] = 0;
        sum2[offset + i] = 0;
    }

    for (int i = idx; i < asian->pathNum; i += blockDim.x * gridDim.x)
    {
        // randNormal(asian, &state, choMatrix, depend + offset, independ + offset);

        for (int j = 0; j < size; j++)
        {
            // double var = depend[offset] + j;
            double var = 1;
            sum[offset + j] += var;
            sum2[offset + j] += var * var;
        }
        sumPerThread += 1;
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
    }
}

double monteCarloGPU(Asian *asian)
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
    double ret = 0;
    double ret2 = 0;

    int size = asian->basketSize;

    cudaMalloc(&option, sizeof(Asian));
    cudaMalloc(&choMatrix, size * size * sizeof(double));

    int totalThread = BLOCK_N * THREAD_N;
    cudaMalloc(&depend, sizeof(double) * size * totalThread);
    cudaMalloc(&independ, sizeof(double) * size * totalThread);
    cudaMalloc(&sum, sizeof(double) * size * totalThread);
    cudaMalloc(&sum2, sizeof(double) * size * totalThread);

    cudaMalloc(&sumOutput, sizeof(double) * size * BLOCK_N);
    cudaMalloc(&sum2Output, sizeof(double) * size * BLOCK_N);
    cudaMemcpy(option, asian, sizeof(Asian), cudaMemcpyHostToDevice);
    cudaMemcpy(choMatrix, asian->choMatrix, size * size * sizeof(double), cudaMemcpyHostToDevice);

    monteCarloOptionKernel<<<BLOCK_N, THREAD_N>>>(option, choMatrix, depend, independ, sum, sumOutput, sum2, sum2Output);
    
    cudaMallocHost(&sumHost, sizeof(double) * size * BLOCK_N);
    cudaMallocHost(&sum2Host, sizeof(double) * size * BLOCK_N);
    cudaMemcpy(sumHost, sumOutput, size *BLOCK_N* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum2Host, sum2Output, size * BLOCK_N* sizeof(double), cudaMemcpyDeviceToHost);
    for (int i=0; i<BLOCK_N; i++){
        ret += sumHost[i*size];
        ret2 += sum2Host[i*size];
    }

    printf("total:%f\n", ret);
    double mean = ret / asian->pathNum;
    
    printf("%f %f\n", mean, ret2 / asian->pathNum - mean * mean);
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
    return 0;
}
