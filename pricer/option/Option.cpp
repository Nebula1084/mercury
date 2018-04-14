#include <option/Option.h>

Asset::Asset()
{
}

Asset::Asset(double price, double volatility, double mean)
{
  this->price = price;
  this->volatility = volatility;
  this->mean = mean;
}

void Asset::setVolatility(double volatility)
{
  this->volatility = volatility;
}

Instrument::Instrument()
{
}

Instrument::Instrument(double maturity, double strike, OptionType type)
{
  this->maturity = maturity;
  this->strike = strike;
  this->type = type;
}