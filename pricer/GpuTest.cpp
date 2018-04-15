#include <asian/Asian.h>
#include <simulate/MonteCarlo.h>
#include <american/American.h>
#include <american/Binomial.h>
#include <european/GeometricEuropean.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

void print(int dim, double *matrix)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            std::cout << matrix[i * dim + j] << " ";
        }

        std::cout << std::endl;
    }
}

void cholesky()
{
    int basketSize = 3;
    double volatility[basketSize] = {0.25, 0.3, 0.1};
    double prices[basketSize] = {5, 5, 5};
    std::cout << "---------Cholesky-----------" << std::endl;
    double corMatrix1[9] = {
        4, 12, -16,
        12, 37, -43,
        -16, -43, 98};
    double strike = 4;
    double maturity = 1;
    double observation = 100;
    int pathNum = 1e5;

    MonteCarlo asin1(basketSize, prices, corMatrix1, volatility, 0.3, maturity,
                     strike, pathNum, observation, CALL);
    auto res = asin1.cholesky();
    print(3, res);
    std::cout << "-------------------------" << std::endl;
    double corMatrix2[4] = {
        1, 0.5,
        0.5, 1};
    MonteCarlo asin2(2, prices, corMatrix2, volatility, 0.3, maturity,
                     strike, pathNum, 10, CALL);
    res = asin2.cholesky();
    print(2, res);
    std::cout << "-------------------------" << std::endl;
    double corMatrix3[9] = {
        1, 0.5, 0.5,
        0.5, 1, 0.5,
        0.5, 0.5, 1};
    MonteCarlo asin3(3, prices, corMatrix3, volatility, 0.1, maturity,
                     strike, pathNum, 10, CALL);
    res = asin3.cholesky();
    print(3, res);
}

void corNormal()
{
    int basketSize = 3;
    std::cout << "----Correlated Normals-----" << std::endl;
    double corMatrix[9] = {
        1, 0.8, 0.9,
        0.8, 1, 0.5,
        0.9, 0.5, 1};
    double volatility[basketSize] = {0.25, 0.3, 0.1};
    double prices[basketSize] = {5, 5, 5};
    double strike = 4;
    double maturity = 1;
    double observation = 100;
    int pathNum = 1e5;

    MonteCarlo asin(3, prices, corMatrix, volatility, 0.3, maturity,
                    strike, pathNum, 10, CALL);

    double sum[basketSize] = {0};
    double sum2[basketSize] = {0};
    double sumX[basketSize * basketSize] = {0};
    curandState state;

    for (int i = 0; i < basketSize; i++)
    {
        for (int j = 0; j < basketSize; j++)
        {
            std::cout << asin.choMatrix[i * basketSize + j] << " ";
        }
        std::cout << std::endl;
    }

    curand_init(1234, 0, 0, &state);
    double *vars = new double[basketSize];
    for (int i = 1; i <= 10000000; i++)
    {
        asin.randNormal(&state, vars);
        for (int j = 0; j < basketSize; j++)
        {
            sum[j] += vars[j];
            sum2[j] += vars[j] * vars[j];
        }
        for (int j = 0; j < basketSize; j++)
            for (int k = 0; k < basketSize; k++)
            {
                sumX[j * basketSize + k] += vars[j] * vars[k];
            }
        if (i % 5000000 == 0)
        {
            printf("%d\n", i);
            double mean[basketSize] = {0};
            double var[basketSize] = {0};
            double cov[basketSize * basketSize] = {0};
            for (int j = 0; j < basketSize; j++)
            {
                mean[j] = sum[j] / i;
                var[j] = sum2[j] / i - mean[j] * mean[j];
                printf("%d: mean = %f, var = %f\n", j, mean[j], var[j]);
            }
            printf("covariance matrix:\n");
            for (int j = 0; j < basketSize; j++)
            {
                for (int k = 0; k < basketSize; k++)
                {
                    cov[j * basketSize + k] = sumX[j * basketSize + k] / i - mean[j] * mean[k];
                    printf("%f ", cov[j * basketSize + k]);
                }
                printf("\n");
            }
            printf("corelation matrix:\n");
            for (int j = 0; j < basketSize; j++)
            {
                for (int k = 0; k < basketSize; k++)
                {
                    printf("%f ", cov[j * basketSize + k] / sqrt(var[j] * var[k]));
                }
                printf("\n");
            }
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

float randFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void monteCarlo()
{
    int PATH_N = 2620;
    int num = 1;
    int basketSize = 2;
    double prices[basketSize] = {5, 5};
    double volatility[basketSize] = {0.25, 0.3};
    double corMatrix[basketSize * basketSize] =
        {1, 0.5,
         0.5, 1};
    double strike = 4;
    double maturity = 1;
    double observation = 100;
    int pathNum = 1e5;

    for (int i = 0; i < num; i++)
    {
        MonteCarlo option(basketSize, prices, corMatrix, volatility, 0.03,
                          maturity, strike, pathNum, observation, CALL);

        double covMatrix[9];
        double expection[3];
        Result callValueCPU = option.simulateCPU(expection, covMatrix);
        printf("Exp : %f \t| Conf: %f\n", callValueCPU.expected, callValueCPU.confidence);
    }
}

void printResult(int basketSize, Result &result, double *covMatrix, double *expection)
{
    printf("Exp : %f \t| Conf: %f\n", result.expected, result.confidence);
    printf("Exp : %f \t| Conf: %f\n", result.geoPayoff, result.confidence);
    printf("covariance matrix:\n");
    for (int j = 0; j < basketSize; j++)
    {
        for (int k = 0; k < basketSize; k++)
        {
            printf("%f ", covMatrix[j * basketSize + k]);
        }
        printf("\n");
    }
    printf("corelation matrix:\n");
    for (int j = 0; j < basketSize; j++)
    {
        for (int k = 0; k < basketSize; k++)
        {
            printf("%f ", covMatrix[j * basketSize + k] / sqrt(covMatrix[j * basketSize + j] * covMatrix[k * basketSize + k]));
        }
        printf("\n");
    }
}

void monteCarloGPU()
{

    double corMatrix[9] = {
        1, 0.8, 0.9,
        0.8, 1, 0.5,
        0.9, 0.5, 1};
    int basketSize = 3;
    double prices[basketSize] = {5, 5, 5};
    double volatility[basketSize] = {0.25, 0.3, 0.1};
    int strike = 4;
    double maturity = 1;
    int pathNum = 100000;

    MonteCarlo option(basketSize, prices, corMatrix, volatility, 0.03,
                      maturity, strike, pathNum, 100, CALL);
    double covMatrix[9];
    double expection[3];
    Result result = option.simulateGPU(expection, covMatrix);
    printResult(basketSize, result, covMatrix, expection);
    result = option.simulateCPU(expection, covMatrix);
    printResult(basketSize, result, covMatrix, expection);
}

void geometricBasket()
{
    bool closedForm = true;
    bool controlVariate = false;
    bool useGpu = false;
    int basketSize = 2;
    double interest = 0.05;
    int pathNum = 1e6;
    Instrument instrument(3, 100, CALL);
    Asset asset[basketSize] = {
        {100, 0.3, interest},
        {100, 0.3, interest}};
    double corMatrix[basketSize * basketSize] = {
        1, 0.5,
        0.5, 1};
    GeometricEuropean option(
        closedForm,
        controlVariate,
        useGpu,
        basketSize,
        interest,
        instrument,
        asset,
        corMatrix,
        pathNum);

    
    std::cout << option.calculate() << std::endl;
    option.closedForm = false;
    option.useGpu = true;
    std::cout << option.calculate() << std::endl;
    option.useGpu = false;
    std::cout << option.calculate() << std::endl;
    controlVariate = false;
}

int main()
{
    // corNormal();
    // monteCarloGPU();
    // monteCarlo();
    geometricBasket();
}
