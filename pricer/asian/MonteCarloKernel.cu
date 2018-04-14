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

__device__ void sumRdx(double *s, double *d, double value)
{
    s[threadIdx.x] = value;
    sumReduce<double, THREAD_N, THREAD_N>(s);
    if (threadIdx.x == 0)
    {
        *d = s[0];
    }
}

__global__ void monteCarloOptionKernel(
    Asian *asian,
    double *choMatrix,
    double *price,
    double *volatility,
    double *drift,
    double *currents,
    double *depend,
    double *independ,
    double *sum,
    double *sumOutput,
    double *sum2,
    double *sum2Output,
    double *sumX,
    double *sumXOutput,
    double *pay,
    double *pay2)
{
    __shared__ double sumThread[THREAD_N];
    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = asian->basketSize;
    int offset = idx * size;
    double dt = 1. / asian->observation;
    double payoff = 0;
    double payPerThread = 0;
    double pay2PerThread = 0;

    curand_init(1230, idx, 0, &state);
    for (int i = 0; i < size; i++)
    {
        sum[offset + i] = 0;
        sum2[offset + i] = 0;
        sumX[offset + i] = 0;
    }

    for (int i = idx; i < asian->pathNum; i += blockDim.x * gridDim.x)
    {

        double mean = 0;
        for (int j = 0; j < size; j++)
        {
            currents[offset + j] = price[j];
        }
        for (int j = 0; j < asian->observation; j++)
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
            for (int k = 0; k < size; k++)
            {
                double growthFactor = drift[k] * exp(volatility[k] * sqrt(dt) * depend[offset + k]);
                currents[offset + k] *= growthFactor;
                mean += currents[offset + k];
            }
        }

        mean /= asian->observation * size;
        payoff = exp(-asian->interest * asian->maturity) * (mean - asian->strike > 0 ? mean - asian->strike : 0);

        payPerThread += payoff;
        pay2PerThread += payoff * payoff;
    }

    sumRdx(sumThread, &pay[blockIdx.x], payPerThread);
    sumRdx(sumThread, &pay2[blockIdx.x], pay2PerThread);

    for (int i = 0; i < size; i++)
    {
        sumRdx(sumThread, &sumOutput[blockIdx.x * size + i], sum[offset + i]);
        sumRdx(sumThread, &sum2Output[blockIdx.x * size + i], sum2[offset + i]);

        for (int j = 0; j < size; j++)
        {
            sumRdx(sumThread, &sumXOutput[blockIdx.x * size * size + i * size + j], sumX[offset * size + i * size + j]);
        }
    }
}

Asian::Value monteCarloGPU(Asian *asian, double *expectation, double *covMatrix)
{
    Asian *option;
    double *choMatrix;
    double *price;
    double *volatility;

    double *drift;
    double *currents;

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

    double *pay;
    double *payHost;
    double *pay2;
    double *pay2Host;

    int size = asian->basketSize;

    cudaMalloc(&option, sizeof(Asian));
    cudaMalloc(&choMatrix, size * size * sizeof(double));
    cudaMalloc(&price, size * sizeof(double));
    cudaMalloc(&volatility, size * sizeof(double));
    cudaMalloc(&drift, size * sizeof(double));

    int totalThread = BLOCK_N * THREAD_N;

    cudaMalloc(&currents, sizeof(double) * size * totalThread);

    cudaMalloc(&depend, sizeof(double) * size * totalThread);
    cudaMalloc(&independ, sizeof(double) * size * totalThread);
    cudaMalloc(&sum, sizeof(double) * size * totalThread);
    cudaMalloc(&sum2, sizeof(double) * size * totalThread);
    cudaMalloc(&sumX, sizeof(double) * size * size * totalThread);

    cudaMalloc(&pay, sizeof(double) * BLOCK_N);
    cudaMalloc(&pay2, sizeof(double) * BLOCK_N);
    cudaMalloc(&sumOutput, sizeof(double) * size * BLOCK_N);
    cudaMalloc(&sum2Output, sizeof(double) * size * BLOCK_N);
    cudaMalloc(&sumXOutput, sizeof(double) * size * size * BLOCK_N);

    cudaMemcpy(option, asian, sizeof(Asian), cudaMemcpyHostToDevice);
    cudaMemcpy(choMatrix, asian->choMatrix, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(price, asian->price, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(volatility, asian->volatility, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(drift, asian->drift, size * sizeof(double), cudaMemcpyHostToDevice);

    monteCarloOptionKernel<<<BLOCK_N, THREAD_N>>>(
        option, choMatrix,
        price, volatility,
        drift, currents,
        depend, independ,
        sum, sumOutput,
        sum2, sum2Output,
        sumX, sumXOutput,
        pay, pay2);

    cudaMallocHost(&payHost, sizeof(double) * BLOCK_N);
    cudaMallocHost(&pay2Host, sizeof(double) * BLOCK_N);
    cudaMallocHost(&sumHost, sizeof(double) * size * BLOCK_N);
    cudaMallocHost(&sum2Host, sizeof(double) * size * BLOCK_N);
    cudaMallocHost(&sumXHost, sizeof(double) * size * size * BLOCK_N);
    cudaMemcpy(payHost, pay, BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pay2Host, pay2, BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sumHost, sumOutput, size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum2Host, sum2Output, size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sumXHost, sumXOutput, size * size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);

    double payRet = 0;
    double pay2Ret = 0;

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
        payRet += payHost[i];
        pay2Ret += pay2Host[i];
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

    Asian::Value ret;

    ret.expected = payRet / (double)asian->pathNum;
    double stdDev = sqrt(((double)asian->pathNum * pay2Ret - payRet * payRet) / ((double)asian->pathNum * (double)(asian->pathNum - 1)));
    ret.confidence = (float)(1.96 * stdDev / sqrt((double)asian->pathNum));

    cudaFreeHost(sumHost);
    cudaFreeHost(sum2Host);
    cudaFreeHost(payHost);
    cudaFreeHost(pay2Host);

    cudaFree(option);
    cudaFree(choMatrix);
    cudaFree(price);
    cudaFree(volatility);

    cudaFree(drift);
    cudaFree(currents);

    cudaFree(depend);
    cudaFree(independ);
    cudaFree(sum);
    cudaFree(sum2);
    cudaFree(sumOutput);
    cudaFree(sum2Output);
    cudaFree(sumXOutput);
    cudaFree(pay);
    cudaFree(pay2);
    return ret;
}
