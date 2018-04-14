#include <european/GeometricEuropean.h>

GeometricEuropean::GeometricEuropean(bool closedForm, bool controlVariate, bool useGpu, int basketSize, double interest,
                                     double repo, Instrument instrument, Asset *asset, double *corMatrix)
    : closedForm(closedForm), controlVariate(controlVariate), useGpu(useGpu),
      BasketEuropean(basketSize, interest, repo, instrument, asset, corMatrix)
{
}

double GeometricEuropean::calculate()
{
    if (closedForm)
        return formulate();
    else
        return simulate();
}

double GeometricEuropean::formulate()
{
    double basketPrice = 1;
    for (int i = 0; i < basketSize; i++)
        basketPrice *= asset[i].price;
    basketPrice = std::pow(basketPrice, 1 / (double)basketSize);

    double sigma = 0;
    for (int i = 0; i < basketSize; i++)
        for (int j = 0; j < basketSize; j++)
            sigma += asset[i].volatility * asset[j].volatility * corMatrix[i * basketSize + j];
    sigma = std::sqrt(sigma) / (double)basketSize;

    double mu = 0;
    for (int i = 0; i < basketSize; i++)
        mu += asset[i].volatility * asset[i].volatility;
    mu = interest - 0.5 * mu / basketPrice + 0.5 * sigma * sigma;

    Asset basketAsset(basketPrice, sigma, mu);
    BlackScholes formula(interest, interest, instrument, basketAsset);
    return formula.calculate();
}

double GeometricEuropean::simulate()
{
    double *volatility = new double[basketSize];
    for (int i = 0; i < basketSize; i++)
        volatility[i] = asset[i].volatility;
    MonteCarlo simulator(basketSize, corMatrix, volatility, interest, 1);
    if (useGpu)
        simulator.simulateGPU();
    else
        simulator.simulateCPU();

    delete[] volatility;
}