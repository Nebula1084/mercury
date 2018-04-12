#ifndef SUM_REDUCTION_CUH
#define SUM_REDUCTION_CUH

template <class T, int SUM_N, int blockSize>
__device__ void sumReduce(T *sum, T *sum2)
{
    for (int stride = SUM_N / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        for (int pos = threadIdx.x; pos < stride; pos += blockSize)
        {
            sum[pos] += sum[pos + stride];
            sum2[pos] += sum2[pos + stride];
        }
    }
}

#endif