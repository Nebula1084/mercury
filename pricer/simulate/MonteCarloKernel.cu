#include <simulate/MonteCarlo.h>
#include <simulate/SumReduction.cuh>
#include <stdio.h>

#define BLOCK_N 256
#define THREAD_N 256

__device__ void randNormal(
    MonteCarlo *plan,
    curandState *state,
    double *choMatrix,
    double *depend,
    double *independ)
{
    int size = plan->basketSize;

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

__device__ double optionValue(MonteCarlo *plan, double value)
{
    return exp(-plan->interest * plan->maturity) * (value > 0 ? value : 0);
}

__global__ void monteCarloOptionKernel(
    MonteCarlo *plan,
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
    double *payArith,
    double *payGeo,
    double *pay2)
{
    __shared__ double sumThread[THREAD_N];
    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = plan->basketSize;
    int offset = idx * size;
    double dt = plan->maturity / plan->observation;
    double arithPayoff = 0;
    double geoPayoff = 0;
    double payArithPerThread = 0;
    double payGeoPerThread = 0;
    double pay2PerThread = 0;

    curand_init(1230, idx, 0, &state);
    for (int i = 0; i < size; i++)
    {
        sum[offset + i] = 0;
        sum2[offset + i] = 0;
        sumX[offset + i] = 0;
    }

    for (int i = idx; i < plan->pathNum; i += blockDim.x * gridDim.x)
    {
        double arithMean = 0;
        double geoMean = 1;

        for (int j = 0; j < size; j++)
        {
            currents[offset + j] = price[j];
        }
        for (int j = 0; j < plan->observation; j++)
        {
            randNormal(plan, &state, choMatrix, depend + offset, independ + offset);
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
                arithMean += currents[offset + k];
                geoMean *= currents[offset + k];
            }
        }

        arithMean /= plan->observation * size;
        geoMean = pow(geoMean, 1 / (double)(plan->observation * size));
        if (plan->type == CALL)
        {
            arithPayoff = optionValue(plan, arithMean - plan->strike);
            geoPayoff = optionValue(plan, geoMean - plan->strike);
        }
        else if (plan->type == PUT)
        {
            arithPayoff = optionValue(plan, plan->strike - arithMean);
            geoPayoff = optionValue(plan, plan->strike - geoMean);
        }

        payArithPerThread += arithPayoff;
        payGeoPerThread += geoPayoff;
        pay2PerThread += arithPayoff * arithPayoff;
    }

    sumRdx(sumThread, &payArith[blockIdx.x], payArithPerThread);
    sumRdx(sumThread, &payGeo[blockIdx.x], payGeoPerThread);
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

Result MonteCarlo::simulateGPU(double *expectation, double *covMatrix)
{
    MonteCarlo *plan;
    double *pChoMatrix;
    double *pPrice;
    double *pVolatility;
    double *pDrift;

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

    double *payArith;
    double *payArithHost;
    double *payGeo;
    double *payGeoHost;
    double *pay2;
    double *pay2Host;

    int size = this->basketSize;

    cudaMalloc(&plan, sizeof(MonteCarlo));
    cudaMalloc(&pChoMatrix, size * size * sizeof(double));
    cudaMalloc(&pPrice, size * sizeof(double));
    cudaMalloc(&pVolatility, size * sizeof(double));
    cudaMalloc(&pDrift, size * sizeof(double));

    int totalThread = BLOCK_N * THREAD_N;

    cudaMalloc(&currents, sizeof(double) * size * totalThread);
    cudaMalloc(&depend, sizeof(double) * size * totalThread);
    cudaMalloc(&independ, sizeof(double) * size * totalThread);
    cudaMalloc(&sum, sizeof(double) * size * totalThread);
    cudaMalloc(&sum2, sizeof(double) * size * totalThread);
    cudaMalloc(&sumX, sizeof(double) * size * size * totalThread);

    cudaMalloc(&payArith, sizeof(double) * BLOCK_N);
    cudaMalloc(&payGeo, sizeof(double) * BLOCK_N);
    cudaMalloc(&pay2, sizeof(double) * BLOCK_N);
    cudaMalloc(&sumOutput, sizeof(double) * size * BLOCK_N);
    cudaMalloc(&sum2Output, sizeof(double) * size * BLOCK_N);
    cudaMalloc(&sumXOutput, sizeof(double) * size * size * BLOCK_N);

    cudaMemcpy(plan, this, sizeof(MonteCarlo), cudaMemcpyHostToDevice);
    cudaMemcpy(pChoMatrix, this->choMatrix, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pPrice, this->price, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pVolatility, this->volatility, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pDrift, this->drift, size * sizeof(double), cudaMemcpyHostToDevice);

    monteCarloOptionKernel<<<BLOCK_N, THREAD_N>>>(
        plan, pChoMatrix,
        pPrice, pVolatility,
        pDrift, currents,
        depend, independ,
        sum, sumOutput,
        sum2, sum2Output,
        sumX, sumXOutput,
        payArith, payGeo, pay2);

    cudaMallocHost(&payArithHost, sizeof(double) * BLOCK_N);
    cudaMallocHost(&payGeoHost, sizeof(double) * BLOCK_N);
    cudaMallocHost(&pay2Host, sizeof(double) * BLOCK_N);
    cudaMallocHost(&sumHost, sizeof(double) * size * BLOCK_N);
    cudaMallocHost(&sum2Host, sizeof(double) * size * BLOCK_N);
    cudaMallocHost(&sumXHost, sizeof(double) * size * size * BLOCK_N);
    cudaMemcpy(payArithHost, payArith, BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(payGeoHost, payGeo, BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pay2Host, pay2, BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sumHost, sumOutput, size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum2Host, sum2Output, size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sumXHost, sumXOutput, size * size * BLOCK_N * sizeof(double), cudaMemcpyDeviceToHost);

    double payArithRet = 0;
    double payGeoRet = 0;
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
        payArithRet += payArithHost[i];
        payGeoRet += payGeoHost[i];
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

    int pathNum = this->pathNum;

    for (int i = 0; i < size; i++)
    {
        expectation[i] /= pathNum;
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            covMatrix[i * size + j] = covMatrix[i * size + j] / pathNum - expectation[i] * expectation[j];
        }
    }

    Result ret;

    ret.expected = payArithRet / (double)pathNum;
    ret.arithPayoff = payArithRet / (double)pathNum;
    ret.geoPayoff = payGeoRet / (double)pathNum;
    double stdDev = sqrt(((double)pathNum * pay2Ret - payArithRet * payArithRet) / ((double)pathNum * (double)(pathNum - 1)));
    ret.confidence = (float)(1.96 * stdDev / sqrt((double)pathNum));

    cudaFreeHost(sumHost);
    cudaFreeHost(sum2Host);
    cudaFreeHost(payArithHost);
    cudaFreeHost(payGeoHost);
    cudaFreeHost(pay2Host);

    cudaFree(plan);
    cudaFree(pChoMatrix);
    cudaFree(pPrice);
    cudaFree(pVolatility);
    cudaFree(pDrift);

    cudaFree(currents);
    cudaFree(depend);
    cudaFree(independ);
    cudaFree(sum);
    cudaFree(sum2);
    cudaFree(sumOutput);
    cudaFree(sum2Output);
    cudaFree(sumXOutput);
    cudaFree(payArith);
    cudaFree(payGeo);
    cudaFree(pay2);
    return ret;
}
