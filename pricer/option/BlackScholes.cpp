#include <option/BlackScholes.h>

BlackScholes::BlackScholes(double interest, double repo, Instrument instrument, Asset asset)
    : interest(interest), repo(repo), instrument(instrument), asset(asset)
{
}

double BlackScholes::calculate()
{
    double S = this->asset.price;
    double sigma = this->asset.volatility;
    double r = this->interest;
    double repo = this->repo;
    double T = this->instrument.maturity;
    double K = this->instrument.strike;
    double timeSqrt = sqrt(T);
    double d1, d2, price;

    d1 = (log(S / K) + (r - repo) * T) / (sigma * timeSqrt) + 0.5 * sigma * timeSqrt;
    d2 = d1 - (sigma * timeSqrt);

    if (this->instrument.type == 1)
    {
        price = S * exp(-repo * T) * normCdf(d1) - K * exp(-r * T) * normCdf(d2);
        return price;
    }
    else if (this->instrument.type == 2)
    {
        price = K * exp(-r * T) * normCdf(-d2) - S * exp(-repo * T) * normCdf(-d1);
        return price;
    }
    else
    {
        return -1;
    }
}

double BlackScholes::vega()
{
    double S = this->asset.price;
    double r = this->interest;
    double repo = this->repo;
    double T = this->instrument.maturity;
    double K = this->instrument.strike;
    double timeSqrt = sqrt(T);
    double sigma = this->asset.volatility;

    double d1 = (log(S / K) + (r - repo) * T) / (sigma * timeSqrt) + 0.5 * sigma * timeSqrt;
    double vega = S * exp(-repo * T) * timeSqrt * normPdf(d1);
    return vega;
}