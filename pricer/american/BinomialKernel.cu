/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// Global types and parameters
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <american/Binomial.h>

//Preprocessed input option data
typedef struct
{
    float S;
    float X;
    float vDt;
    float puByDf;
    float pdByDf;
} __TOptionData;
static __constant__ __TOptionData d_OptionData[MAX_OPTIONS];
static __device__           float d_CallValue[MAX_OPTIONS];



////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
    float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
    return (d > 0.0F) ? d : 0.0F;
}

////////////////////////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////////////////////////
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

__global__ void binomialOptionsKernel()
{
    __shared__ float call_exchange[THREADBLOCK_SIZE + 1];

    const int     tid = threadIdx.x;
    const float      S = d_OptionData[blockIdx.x].S;
    const float      X = d_OptionData[blockIdx.x].X;
    const float    vDt = d_OptionData[blockIdx.x].vDt;
    const float puByDf = d_OptionData[blockIdx.x].puByDf;
    const float pdByDf = d_OptionData[blockIdx.x].pdByDf;

    float call[ELEMS_PER_THREAD + 1];
    #pragma unroll
    for(int i = 0; i < ELEMS_PER_THREAD; ++i)
        call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

    if (tid == 0)
        call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

    int final_it = max(0, tid * ELEMS_PER_THREAD - 1);

    #pragma unroll 16
    for(int i = NUM_STEPS; i > 0; --i)
    {
        call_exchange[tid] = call[0];
        __syncthreads();
        call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
        __syncthreads();

        if (i > final_it)
        {
           #pragma unroll
           for(int j = 0; j < ELEMS_PER_THREAD; ++j)
              call[j] = puByDf * call[j + 1] + pdByDf * call[j];
        }
    }

    if (tid == 0)
    {
        d_CallValue[blockIdx.x] = call[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU binomialOptions
////////////////////////////////////////////////////////////////////////////////
void binomialOptionsGPU(
    float *callValue,
    American  *americans,
    int num
)
{
    __TOptionData h_OptionData[MAX_OPTIONS];

    for (int i = 0; i < num; i++)
    {
        const float      T = americans[i].T;
        const float      R = americans[i].R;
        const float      V = americans[i].V;

        const float     dt = T / (float)NUM_STEPS;
        const float    vDt = V * sqrt(dt);
        const float    rDt = R * dt;
        //Per-step interest and discount factors
        const float     If = exp(rDt);
        const float     Df = exp(-rDt);
        //Values and pseudoprobabilities of upward and downward moves
        const float      u = exp(vDt);
        const float      d = exp(-vDt);
        const float     pu = (If - d) / (u - d);
        const float     pd = (float)1.0 - pu;
        const float puByDf = pu * Df;
        const float pdByDf = pd * Df;

        h_OptionData[i].S      = (float)americans[i].S;
        h_OptionData[i].X      = (float)americans[i].X;
        h_OptionData[i].vDt    = (float)vDt;
        h_OptionData[i].puByDf = (float)puByDf;
        h_OptionData[i].pdByDf = (float)pdByDf;
    }

    cudaMemcpyToSymbol(d_OptionData, h_OptionData, num * sizeof(__TOptionData));
    binomialOptionsKernel<<<num, THREADBLOCK_SIZE>>>();
    cudaMemcpyFromSymbol(callValue, d_CallValue, num *sizeof(float));
}
