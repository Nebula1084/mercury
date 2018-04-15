#include <simulate/MonteCarlo.h>

MonteCarlo::MonteCarlo(int basketSize, double *price, double *corMatrix, double *volatility,
                       double interest, double maturity, double strike,
                       int pathNum, int observation, OptionType type)
    : basketSize(basketSize), price(price), corMatrix(corMatrix), volatility(volatility), interest(interest),
      maturity(maturity), strike(strike), pathNum(pathNum), observation(observation), type(type)
{
    this->choMatrix = cholesky();
    this->drift = new double[basketSize];

    double dt = maturity / observation;
    for (int i = 0; i < basketSize; i++)
    {
        this->drift[i] = exp((interest - 0.5 * volatility[i] * volatility[i]) * dt);
        // std::cout << drift[i] << std::endl;
    }
}

double *MonteCarlo::cholesky()
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

void MonteCarlo::randNormal(curandState *state, double *dependNormals)
{
    double *independNormals = new double[basketSize];

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
}

double MonteCarlo::optionValue(double value)
{
    return exp(-interest * maturity) * (value > 0 ? value : 0);
}

Result MonteCarlo::simulateCPU(double *expectation, double *covMatrix)
{
    double sum2 = 0, payArith = 0, payGeo = 0;
    double *normals = new double[basketSize];
    double *currents = new double[basketSize];

    curandState state;
    curand_init(2230, 0, 0, &state);

    double dt = maturity / observation;
    double arithPayoff = 0;
    double geoPayoff = 0;

    for (int i = 0; i < pathNum; i++)
    {
        double arithMean = 0;
        double geoMean = 1;
        for (int j = 0; j < basketSize; j++)
        {
            currents[j] = price[j];
        }
        for (int j = 0; j < observation; j++)
        {
            randNormal(&state, normals);
            for (int k = 0; k < basketSize; k++)
            {
                double growthFactor = drift[k] * exp(volatility[k] * sqrt(dt) * normals[k]);
                currents[k] *= growthFactor;
                arithMean += currents[k];
                geoMean *= currents[k];
                // std::cout << j << " " << k << " " << currents[k] << std::endl;
            }
        }

        arithMean /= observation * basketSize;
        geoMean = std::pow(geoMean, 1 / (double)(observation * basketSize));
        if (this->type == CALL)
        {
            arithPayoff = optionValue(arithMean - strike);
            geoPayoff = optionValue(geoMean - strike);
        }
        else if (this->type == PUT)
        {
            arithPayoff = optionValue(strike - arithMean);
            geoPayoff = optionValue(strike - geoMean);
        }
        // std::cout << geoMean - strike << std::endl;
        payArith += arithPayoff;
        payGeo += geoPayoff;
        sum2 += arithPayoff * arithPayoff;
    }

    delete[] normals;
    delete[] currents;
    Result ret;

    ret.expected = payArith / (double)pathNum;
    ret.arithPayoff = payArith / (double)pathNum;
    ret.geoPayoff = payGeo / (double)pathNum;
    double stdDev = sqrt(((double)pathNum * sum2 - payArith * payArith) / ((double)pathNum * (double)(pathNum - 1)));
    ret.confidence = (float)(1.96 * stdDev / sqrt((double)pathNum));
    return ret;
}
