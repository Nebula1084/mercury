#include <european/Volatility.h>

Volatility::Volatility(Protocol *buff)
    : Volatility(buff->interest, buff->repo, buff->instrument, buff->premium, buff->asset.price)
{
}

Volatility::Volatility(double interest, double repo, Instrument instrument, double premium, double price)
    : interest(interest), repo(repo), instrument(instrument), premium(premium), asset(Asset(price, -1))
{
}

Result Volatility::calculate()
{
    double S = this->asset.price;
    double r = this->interest;
    double repo = this->repo;
    double T = this->instrument.maturity;
    double K = this->instrument.strike;
    double premium = this->premium;

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
        // std::cout << sigma << std::endl;
        formula.asset.setVolatility(sigma);
        value = formula.calculate();
        vega = formula.vega();
        increment = (value - premium) / vega;
        sigma = sigma - increment;
        sigmaDiff = std::abs(increment);
        n++;
    }

    this->asset.setVolatility(sigma);

    Result result;
    result.conf = -1;

    if (std::abs(value - premium) < 1e-4)
        result.mean = sigma;
    else
        result.mean = -1;
    return result;
}