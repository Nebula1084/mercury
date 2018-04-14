#include <option/Option.h>

Asset::Asset()
{
}

Asset::Asset(float S, float sigma)
{
  this->price = S;
  this->volatility = sigma;
}

void Asset::setVolatility(float sigma)
{
  this->volatility = sigma;
}


Instrument::Instrument()
{
}

Instrument::Instrument(float maturity, float strike, OptionType type)
{
  this->maturity = maturity;
  this->strike = strike;
  this->type = type;
}