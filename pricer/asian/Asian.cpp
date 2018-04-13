#include <asian/Asian.h>

Asian::Asian()
{
}

Asian::Asian(int basketSize, double *corMatrix)
    : basketSize(basketSize), corMatrix(corMatrix)
{
    this->choMatrix = cholesky();
}

double *Asian::cholesky()
{

    double *lMatrix = new double[basketSize * basketSize];

    for (int i = 0; i < basketSize; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            lMatrix[i * basketSize + j] = corMatrix[i * basketSize + j];

            for (int k = 0; k < j; k++)
                lMatrix[i * basketSize + j] -= lMatrix[i * basketSize + k] * lMatrix[j * basketSize + k];
            if (i == j)
                lMatrix[i * basketSize + j] = std::sqrt(lMatrix[i * basketSize + j]);
            else
                lMatrix[i * basketSize + j] = lMatrix[i * basketSize + j] / lMatrix[j * basketSize + j];
        }
        for (int j = i + 1; j < basketSize; j++)
        {
            lMatrix[i * basketSize + j] = 0;
        }
    }

    return lMatrix;
}

double *Asian::randNormal(curandState *state)
{
    double *independNormals = new double[basketSize];
    double *dependNormals = new double[basketSize];

    for (int i = 0; i < basketSize; i++)
    {
        independNormals[i] = curand_normal(state);
    }

    for (int i = 0; i < basketSize; i++)
    {
        double corNormal = 0;
        for (int j = 0; j < basketSize; j++)
        {
            corNormal += independNormals[j] * choMatrix[i * basketSize + j];
        }
        dependNormals[i] = corNormal;
    }

    delete[] independNormals;
    return dependNormals;
}

static double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}

Asian::Value Asian::monteCarloCPU(int pathN)
{
    const double MuByT = (R - 0.5 * V * V) * T;
    const double VBySqrtT = V * sqrt(T);
    float *samples;
    curandGenerator_t cudaGen;
    unsigned long long seed = 1234ULL;

    curandCreateGeneratorHost(&cudaGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(cudaGen, seed);
    samples = new float[pathN];
    curandGenerateNormal(cudaGen, samples, pathN, 0.0, 1.0);

    double sum = 0, sum2 = 0;

    for (int pos = 0; pos < pathN; pos++)
    {

        double sample = samples[pos];
        double callValue = endCallValue(S, X, sample, MuByT, VBySqrtT);
        sum += callValue;
        sum2 += callValue * callValue;
    }

    delete[] samples;

    curandDestroyGenerator(cudaGen);

    Value ret;

    ret.expected = (float)(exp(-R * T) * sum / (double)pathN);
    double stdDev = sqrt(((double)pathN * sum2 - sum * sum) / ((double)pathN * (double)(pathN - 1)));
    ret.confidence = (float)(exp(-R * T) * 1.96 * stdDev / sqrt((double)pathN));
    return ret;
}