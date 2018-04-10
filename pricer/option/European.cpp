#include <option/European.h>

European::European()
{
}

European::European(float r, float repo, Instrument instrument, Asset asset)
{
  this->interest = r;
  this->repo = repo;
  this->instrument = instrument;
  this->asset = asset;
}

float European::calculate()
{
    float S = this->asset.price;
    float sigma = this->asset.volatility;
    float r = this->interest;
    float repo = this->repo;
    float T = this->instrument.maturity;
    float K = this->instrument.strike;    
    float time_sqrt = sqrt(T);
    float d1, d2, price;
    
    d1 = (log(S/K)+(r-repo)*T)/(sigma*time_sqrt) + 0.5*sigma*time_sqrt;
    d2 = d1 - (sigma*time_sqrt);

    if (this->instrument.type == 1)
    {    
         price = S*exp(-repo*T)*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2);
         return price;
    }
    else if (this->instrument.type == 2)
    {
        price = K*exp(-r*T)*norm_cdf(-d2) - S*exp(-repo*T)*norm_cdf(-d1);
        return price;
    } else
    {
        return -1;
    }

}