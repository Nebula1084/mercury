#include <option/Volatility.h>
#include <option/European.h>
#include <iostream>

float vegaValue(float S, float K, float T, float r, float repo, float sigma)
{
    float time_sqrt = sqrt(T);
    float d1 = (log(S / K) + (r - repo) * T) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt;
    float vega = S * exp(-repo * T) * time_sqrt * norm_pdf(d1);
    return vega;
}

Volatility::Volatility()
{
}

Volatility::Volatility(float r, float repo, Instrument instrument, float price, float S)
{
    this->interest = r;
    this->repo = repo;
    this->instrument = instrument;
    this->price = price;
    this->asset.price = S;
}

float Volatility::calculate()
{
    float S = this->asset.price;
    float r = this->interest;
    float repo = this->repo;
    float T = this->instrument.maturity;
    float K = this->instrument.strike;
    float premium = this->price;

    float sigmahat = sqrt(2 * abs((log(S / K) + (r - repo) * T) / T));

    float tol = 1e-8;
    float sigma = sigmahat;
    float sigmadiff = 1;
    int n = 1;
    int nmax = 100;
    float value, vega, increment;

    //construct an European Object
    Asset asset(S, sigma);
    European european(r, repo, this->instrument, asset);

    //upper bound and lower bound

    while (sigmadiff >= tol && n < nmax)
    {
        european.asset.setVolatility(sigma);
        value = european.calculate();
        vega = vegaValue(S, K, T, r, repo, sigma);
        increment = (value - premium) / vega;
        sigma = sigma - increment;
        n++;
        sigmadiff = abs(increment);
    }

    this->asset.setVolatility(sigma);

    if (abs(value - premium) < 1e-4)
        return sigma;
    else
        return -1;
}