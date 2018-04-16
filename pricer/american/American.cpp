#include <american/American.h>

American::American(bool useGpu, double interest, Asset asset, Instrument instrument, int step)
    : useGpu(useGpu), interest(interest), asset(asset), instrument(instrument), step(step)
{
}

double American::optionValue(int i, int j)
{
    double strike = instrument.strike;
    double d = asset.price * std::exp(vDt * (2.0 * i - j));
    if (instrument.type == CALL)
        d = d - strike;
    else if (instrument.type == PUT)
        d = strike - d;
    return (d > 0) ? d : 0;
}

Result American::calculate()
{
    Result result;
    preprocess();
    if (useGpu)
        result.mean = binomialGPU();
    else
        result.mean = binomialCPU();
    result.conf = -1;
    return result;
}

void American::preprocess()
{
    dt = instrument.maturity / (double)step;
    vDt = asset.volatility * sqrt(dt);
    rDt = interest * dt;

    If = exp(rDt);
    Df = exp(-rDt);

    u = exp(vDt);
    d = exp(-vDt);
    pu = (If - d) / (u - d);
    pd = 1.0 - pu;
    puByDf = pu * Df;
    pdByDf = pd * Df;
}

double American::binomialCPU()
{

    double expiryValue[step + 1];

    for (int i = 0; i <= step; i++)
        expiryValue[i] = optionValue(i, step);

    for (int i = step; i > 0; i--)
        for (int j = 0; j < i; j++)
        {
            double exercise = optionValue(j, i - 1);
            double estimate = puByDf * expiryValue[j + 1] + pdByDf * expiryValue[j];
            expiryValue[j] = exercise > estimate ? exercise : estimate;
        }

    return expiryValue[0];
}
