#include <european/GeometricEuropean.h>

GeometricEuropean::GeometricEuropean(bool closedForm, bool useGpu, int basketSize, double interest,
                                     Instrument instrument, Asset *asset, double *corMatrix, int pathNum)
    : closedForm(closedForm),
      BasketEuropean(basketSize, interest, 0, instrument, asset, corMatrix, useGpu, pathNum)
{
}

double GeometricEuropean::calculate()
{
    if (closedForm)
        return formulate();
    else
        return simulate().geoPayoff;
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
    mu = interest - 0.5 * mu / basketSize + 0.5 * sigma * sigma;

    Asset basketAsset(basketPrice, sigma, mu);
    BlackScholes formula(interest, 0, instrument, basketAsset);
    return formula.calculate();
}