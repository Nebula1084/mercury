#include <asian/Asian.h>
#include <american/American.h>
#include <american/Binomial.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

void print(std::vector<std::vector<float>> matrix)
{
    for (auto &col : matrix)
    {
        for (auto &val : col)
        {
            std::cout << val << " ";
        }

        std::cout << std::endl;
    }
}

void cholesky()
{
    std::cout << "---------Cholesky-----------" << std::endl;
    std::vector<std::vector<float>> corMatrix1 = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};

    Asian asin1(corMatrix1);
    auto res = asin1.cholesky();
    print(res);
    std::cout << "-------------------------" << std::endl;
    std::vector<std::vector<float>> corMatrix2 = {
        {1, 0.5},
        {0.5, 1}};
    Asian asin2(corMatrix2);
    res = asin2.cholesky();
    print(res);
    std::cout << "-------------------------" << std::endl;
    std::vector<std::vector<float>> corMatrix3 = {
        {1, 0.5, 0.5},
        {0.5, 1, 0.5},
        {0.5, 0.5, 1}};
    Asian asin3(corMatrix3);
    res = asin3.cholesky();
    print(res);
}

void corNormal()
{
    std::cout << "----Correlated Normals-----" << std::endl;
    std::vector<std::vector<float>> corMatrix = {
        {1, 0.5, 0.5},
        {0.5, 1, 0.5},
        {0.5, 0.5, 1}};
    Asian asin(corMatrix);
    int basketSize = 3;
    std::vector<float> sum(basketSize);
    for (int i = 1; i <= 1000000; i++)
    {
        auto vars = asin.randNormal();
        for (int j = 0; j < basketSize; j++)
            sum[j] += vars[j];
        if (i % 100000 == 0)
            for (int j = 0; j < basketSize; j++)
            {
                std::cout << i << ":" << sum[j] / i << std::endl;
            }
    }
}

float randData(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return ((float)1.0 - t) * low + t * high;
}

void binomial()
{

    American options[MAX_OPTIONS];
    float callBS[MAX_OPTIONS];
    float callCPU[MAX_OPTIONS];
    float callGPU[MAX_OPTIONS];

    std::cout << "Generating input data..."
              << std::endl;

    for (int i = 0; i < MAX_OPTIONS; i++)
    {
        options[i].S = randData(5.0f, 30.0f);
        options[i].X = randData(1.0f, 100.0f);
        options[i].T = randData(0.25f, 10.0f);
        options[i].R = 0.06f;
        options[i].V = 0.10f;
        callBS[i] = options[i].BlackScholesCall();
    }

    printf("Running GPU binomial tree...\n");
    cudaDeviceSynchronize();

    binomialOptionsGPU(callGPU, options, MAX_OPTIONS);

    cudaDeviceSynchronize();
    float sumDelta = 0;
    float sumRef = 0;
    printf("GPU binomial vs. Black-Scholes\n");

    for (int i = 0; i < MAX_OPTIONS; i++)
    {
        sumDelta += fabs(callBS[i] - callGPU[i]);
        sumRef += fabs(callBS[i]);
    }

    printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
    printf("Avg. diff: %E\n", (double)(sumDelta / (float)MAX_OPTIONS));

    std::cout << "Running CPU binomial tree..." << std::endl;
    for (int i = 0; i < MAX_OPTIONS; i++)
    {
        callCPU[i] = options[i].binomialOptionsCPU();
    }

    sumDelta = 0;
    sumRef = 0;
    std::cout << "CPU binomial vs. Black-Scholes" << std::endl;

    for (int i = 0; i < MAX_OPTIONS; i++)
    {
        sumDelta += fabs(callBS[i] - callCPU[i]);
        sumRef += fabs(callBS[i]);
    }

    printf("L1 norm: %E\n", sumDelta / sumRef);
    printf("Avg. diff: %E\n", (double)(sumDelta / (float)MAX_OPTIONS));

    printf("CPU binomial vs. GPU binomial\n");
    sumDelta = 0;
    sumRef = 0;

    for (int i = 0; i < MAX_OPTIONS; i++)
    {
        sumDelta += fabs(callGPU[i] - callCPU[i]);
        sumRef += callCPU[i];
    }

    printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
    printf("Avg. diff: %E\n", (double)(sumDelta / (float)MAX_OPTIONS));
}

int main()
{
    binomial();
}
