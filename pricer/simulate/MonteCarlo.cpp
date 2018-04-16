#include <simulate/MonteCarlo.h>

MonteCarlo::MonteCarlo(int basketSize, double *price, double *corMatrix, double *volatility,
                       double interest, double maturity, double strike, int pathNum,
                       int observation, OptionType type, bool isGeo)
    : basketSize(basketSize), price(price), corMatrix(corMatrix), volatility(volatility), interest(interest),
      maturity(maturity), strike(strike), pathNum(pathNum), observation(observation), type(type),
      isGeo(isGeo), controlVariate(false), geoExp(0)
{
    this->choMatrix = cholesky();
    this->drift = new double[basketSize];

    double dt = maturity / observation;
    for (int i = 0; i < basketSize; i++)
    {
        this->drift[i] = exp((interest - 0.5 * volatility[i] * volatility[i]) * dt);
    }
}

void MonteCarlo::setControlVariate(bool control, double geoExp)
{
    this->controlVariate = control;
    this->geoExp = geoExp;
}

double MonteCarlo::confidence(double std)
{
    return 1.96 * std / sqrt((double)pathNum);
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

void MonteCarlo::statistic(double *values, double &mean, double &std)
{
    double sum = 0, sum2 = 0;
    for (int i = 0; i < pathNum; i++)
    {
        double v = values[i];
        sum += v;
        sum2 += v * v;
    }
    mean = sum / pathNum;
    std = std::sqrt(sum2 / pathNum - mean * mean);
}

double MonteCarlo::covariance(double *arith, double *geo, double arithMean, double geoMean)
{
    double sum = 0;
    for (int i = 0; i < pathNum; i++)
        sum += arith[i] * geo[i];
    return sum / pathNum - arithMean * geoMean;
}

void MonteCarlo::variationReduce(double *dst, double *arithPayoff, double *geoPayoff, double theta)
{
    for (int i = 0; i < pathNum; i++)
        dst[i] = arithPayoff[i] + theta * (geoExp - geoPayoff[i]);
}

Result MonteCarlo::simulateCPU(double *expectation, double *covMatrix)
{
    double normals[basketSize];
    double currents[basketSize];

    curandState state;
    curand_init(2230, 0, 0, &state);

    double dt = maturity / observation;
    double *arithPayoff = new double[pathNum];
    double *geoPayoff = new double[pathNum];

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
            }
        }

        arithMean /= observation * basketSize;
        geoMean = std::pow(geoMean, 1 / (double)(observation * basketSize));
        if (this->type == CALL)
        {
            arithPayoff[i] = optionValue(arithMean - strike);
            geoPayoff[i] = optionValue(geoMean - strike);
        }
        else if (this->type == PUT)
        {
            arithPayoff[i] = optionValue(strike - arithMean);
            geoPayoff[i] = optionValue(strike - geoMean);
        }
    }

    Result ret;
    double aMean, gMean, aStd, gStd;

    statistic(arithPayoff, aMean, aStd);
    statistic(geoPayoff, gMean, gStd);

    if (isGeo)
    {
        ret.mean = gMean;
        ret.conf = confidence(gStd);
    }
    else
    {
        if (controlVariate)
        {
            double cov = covariance(arithPayoff, geoPayoff, aMean, gMean);
            double theta = cov / (gStd * gStd);
            double *newArith = new double[pathNum];
            variationReduce(newArith, arithPayoff, geoPayoff, theta);
            statistic(newArith, aMean, aStd);
            delete[] newArith;
        }
        ret.mean = aMean;
        ret.conf = confidence(aStd);
    }

    // ret.confidence = (float)(1.96 * stdDev / sqrt((double)pathNum));
    delete[] arithPayoff;
    delete[] geoPayoff;
    return ret;
}
