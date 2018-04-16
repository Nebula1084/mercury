#include <simulate/MonteCarlo.h>

#include <asian/Asian.h>
#include <asian/GeometricAsian.h>
#include <asian/ArithmeticAsian.h>

#include <american/American.h>

#include <european/European.h>
#include <european/GeometricEuropean.h>
#include <european/ArithmeticEuropean.h>

#include <vector>
#include <iostream>
#include <stdio.h>

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
                     strike, pathNum, observation, CALL, false);
    auto res = asin1.cholesky();
    print(3, res);
    std::cout << "-------------------------" << std::endl;
    double corMatrix2[4] = {
        1, 0.5,
        0.5, 1};
    MonteCarlo asin2(2, prices, corMatrix2, volatility, 0.3, maturity,
                     strike, pathNum, 10, CALL, false);
    res = asin2.cholesky();
    print(2, res);
    std::cout << "-------------------------" << std::endl;
    double corMatrix3[9] = {
        1, 0.5, 0.5,
        0.5, 1, 0.5,
        0.5, 0.5, 1};
    MonteCarlo asin3(3, prices, corMatrix3, volatility, 0.1, maturity,
                     strike, pathNum, 10, CALL, false);
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
                    strike, pathNum, 10, CALL, false);

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

    double interest = 0.06f;
    double volatility = 0.10f;
    double price = 100;
    double strike = 100;
    double maturity = 3;
    int step = 5000;
    Asset asset(price, volatility);
    Instrument instrument(maturity, strike, CALL);

    European european(interest, 0, instrument, asset);
    std::cout << european.calculate() << std::endl;
    european.instrument.type = PUT;
    std::cout << european.calculate() << std::endl;

    American option(false, interest, asset, instrument, step);

    option.useGpu = true;
    std::cout << option.calculate() << std::endl;
    option.useGpu = false;
    std::cout << option.calculate() << std::endl;

    option.instrument.type = PUT;
    option.useGpu = true;
    std::cout << option.calculate() << std::endl;
    option.useGpu = false;
    std::cout << option.calculate() << std::endl;
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
                          maturity, strike, pathNum, observation, CALL, false);

        double covMatrix[9];
        double expection[3];
        Result callValueCPU = option.simulateCPU(expection, covMatrix);

        printf("Arith : %f \t| Conf: %f\n", callValueCPU.mean, callValueCPU.conf);
    }
}

void printResult(int basketSize, Result &result, double *covMatrix, double *expection)
{
    printf("Mean : %f \t| Conf: %f\n", result.mean, result.conf);
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
                      maturity, strike, pathNum, 100, CALL, false);
    double covMatrix[9];
    double expection[3];
    Result result = option.simulateGPU(expection, covMatrix);
    printResult(basketSize, result, covMatrix, expection);
    // result = option.simulateCPU(expection, covMatrix);
    // printResult(basketSize, result, covMatrix, expection);
}

void geometricBasket()
{
    bool closedForm = true;
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
        useGpu,
        basketSize,
        interest,
        instrument,
        asset,
        corMatrix,
        pathNum);

    std::cout << "----Geometric Basket-----" << std::endl;
    std::cout << option.calculate() << std::endl;
    option.closedForm = false;
    std::cout << "GPU" << std::endl;
    option.useGpu = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "CPU" << std::endl;
    option.useGpu = false;
    std::cout << option.calculate() << std::endl;
}

void arithmeticBasket()
{
    bool controlVariate = false;
    bool useGpu = false;
    int basketSize = 2;
    double interest = 0.05;
    int pathNum = 1e5;
    Instrument instrument(3, 100, CALL);
    Asset asset[basketSize] = {
        {100, 0.3, interest},
        {100, 0.3, interest}};
    double corMatrix[basketSize * basketSize] = {
        1, 0.5,
        0.5, 1};
    ArithmeticEuropean option(
        controlVariate,
        useGpu,
        basketSize,
        interest,
        instrument,
        asset,
        corMatrix,
        pathNum);
    std::cout << "---Arithmetic Basket CALL--" << std::endl;
    std::cout << "GPU" << std::endl;
    option.useGpu = true;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "CPU" << std::endl;
    option.useGpu = false;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "---Arithmetic Basket PUT---" << std::endl;
    option.instrument.type = PUT;
    std::cout << "GPU" << std::endl;
    option.useGpu = true;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "CPU" << std::endl;
    option.useGpu = false;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
}

void geometricAsian()
{
    bool closedForm = true;
    bool useGpu = false;
    int basketSize = 2;
    double interest = 0.05;
    int pathNum = 1e5;
    int observation = 50;

    Instrument instrument(3, 100, CALL);
    Asset asset(100, 0.3, interest);

    GeometricAsian option(
        closedForm,
        asset,
        interest,
        instrument,
        useGpu,
        pathNum,
        observation);
    std::cout << "----Geometric Basket-----" << std::endl;
    std::cout << option.calculate() << std::endl;
    option.closedForm = false;
    std::cout << "GPU" << std::endl;
    option.useGpu = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "CPU" << std::endl;
    option.useGpu = false;
    std::cout << option.calculate() << std::endl;
}

void arithmeticAsian()
{
    bool controlVariate = true;
    bool useGpu = false;
    int basketSize = 2;
    double interest = 0.05;
    int pathNum = 1e5;
    int observation = 50;

    Instrument instrument(3, 100, CALL);
    Asset asset(100, 0.3, interest);

    ArithmeticAsian option(
        controlVariate,
        asset,
        interest,
        instrument,
        useGpu,
        pathNum,
        observation);

    std::cout << "---Arithmetic Asian CALL--" << std::endl;
    std::cout << "GPU" << std::endl;
    option.useGpu = true;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "CPU" << std::endl;
    option.useGpu = false;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "---Arithmetic Asian PUT---" << std::endl;
    option.instrument.type = PUT;
    std::cout << "GPU" << std::endl;
    option.useGpu = true;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
    std::cout << "CPU" << std::endl;
    option.useGpu = false;
    option.controlVariate = false;
    std::cout << option.calculate() << std::endl;
    option.controlVariate = true;
    std::cout << option.calculate() << std::endl;
}

int main()
{
    binomial();
    corNormal();
    monteCarloGPU();
    monteCarlo();
    geometricBasket();
    arithmeticBasket();
    geometricAsian();
    arithmeticAsian();
}
