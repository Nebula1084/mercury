#include <asian/Asian.h>

Asian::Asian(Protocol *buff)
    : Asian(buff->asset, buff->interest, buff->instrument, buff->useGpu, buff->pathNum, buff->step)
{
}

Asian::Asian(Asset asset, double interest, Instrument instrument, bool useGpu, int pathNum, int observation)
    : asset(asset), interest(interest), instrument(instrument), observation(observation),
      useGpu(useGpu), pathNum(pathNum)
{
}

double Asian::formulate()
{
    double v = asset.volatility;
    double n = observation;
    double sigma = v * std::sqrt((n + 1) * (2 * n + 1) / (6 * n * n));
    double mu = (interest - 0.5 * v * v) * (n + 1) / (2 * n) + 0.5 * sigma * sigma;

    Asset asianAsset(asset.price, sigma, mu);
    BlackScholes formula(interest, 0, instrument, asianAsset);
    return formula.calculate();
}

Result Asian::simulate(bool isGeo, bool control)
{
    int basketSize = 1;
    double volatility[basketSize];
    double price[basketSize];
    double covMatrix[basketSize * basketSize];
    double expectation[basketSize];
    double corMatrix[1] = {1};

    price[0] = asset.price;
    volatility[0] = asset.volatility;
    MonteCarlo simulator(basketSize, price, corMatrix, volatility, interest, instrument.maturity,
                         instrument.strike, pathNum, observation, instrument.type, isGeo);
    if (control)
        simulator.setControlVariate(control, formulate());
    if (useGpu)
        return simulator.simulateGPU(expectation, covMatrix);
    else
        return simulator.simulateCPU(expectation, covMatrix);
}