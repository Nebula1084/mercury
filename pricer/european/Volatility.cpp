#include <european/Volatility.h>

Volatility::Volatility()
{
}

Volatility::Volatility(double r, double repo, Instrument instrument, double price, double S)
{
    this->interest = r;
    this->repo = repo;
    this->instrument = instrument;
    this->price = price;
    this->asset.price = S;
}

double Volatility::calculate()
{
    double S = this->asset.price;
    double r = this->interest;
    double repo = this->repo;
    double T = this->instrument.maturity;
    double K = this->instrument.strike;
    double premium = this->price;

    double sigmaHat = std::sqrt(2 * std::abs((log(S / K) + (r - repo) * T) / T));

    double tol = 1e-8;
    double sigma = sigmaHat;
    double sigmaDiff = 1;
    int n = 1;
    int nmax = 100;
    double value, vega, increment;


    //construct an European Object
    Asset asset(S, sigma, interest);
    BlackScholes formula(r, repo, this->instrument, asset);

    //upper bound and lower bound

    while (sigmaDiff >= tol && n < nmax)
    {
        formula.asset.setVolatility(sigma);
        value = formula.calculate();
        vega = formula.vega();
        increment = (value - premium) / vega;
        sigma = sigma - increment;
        sigmaDiff = std::abs(increment);
        n++;
    }

    this->asset.setVolatility(sigma);

    if (std::abs(value - premium) < 1e-4)
    {

        return sigma;
    }
    else
        return -1;
}