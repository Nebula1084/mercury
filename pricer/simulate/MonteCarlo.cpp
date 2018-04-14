#include <simulate/MonteCarlo.h>

MonteCarlo::MonteCarlo(int basketSize, double *corMatrix, double *volatility, double interest, int observation)
    : basketSize(basketSize), corMatrix(corMatrix), volatility(volatility), interest(interest), observation(observation)
{
    this->choMatrix = cholesky();
    this->drift = new double[basketSize];

    double dt = 1. / observation;
    for (int i = 0; i < basketSize; i++)
    {
        this->drift[i] = exp((interest - 0.5 * volatility[i] * volatility[i]) * dt);
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

Value MonteCarlo::simulateCPU(double *expectation, double *covMatrix)
{
    double sum = 0, sum2 = 0;
    double *normals = new double[basketSize];
    double *currents = new double[basketSize];

    curandState state;
    curand_init(2230, 0, 0, &state);
    double mean;

    double dt = 1. / observation;
    double payoff;

    for (int i = 0; i < pathNum; i++)
    {
        mean = 0;
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
                mean += currents[k];
            }
        }

        mean /= observation * basketSize;
        payoff = exp(-interest * maturity) * (mean - strike > 0 ? mean - strike : 0);

        sum += payoff;
        sum2 += payoff * payoff;
    }

    delete[] normals;
    delete[] currents;
    Value ret;

    ret.expected = sum / (double)pathNum;
    double stdDev = sqrt(((double)pathNum * sum2 - sum * sum) / ((double)pathNum * (double)(pathNum - 1)));
    ret.confidence = (float)(1.96 * stdDev / sqrt((double)pathNum));
    return ret;
}
