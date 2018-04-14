#ifndef OPTION_H
#define OPTION_H

#include <option/Norm.h>

typedef char OptionType;

const OptionType CALL = 1;
const OptionType PUT = 2;

typedef char Operation;

const Operation EUROPEAN = 1;
const Operation VOLATILITY = 2;
const Operation AMERICAN = 3;

class Asset
{
private:
  friend class European;
  friend class Volatility;
  float price;
  float volatility;

public:
  void setVolatility(float sigma);
  Asset();
  Asset(float S, float sigma);

} __attribute__((packed));

class Instrument
{
private:
  friend class European;
  friend class Volatility;
  float maturity;
  float strike;
  OptionType type;

public:
  Instrument();
  Instrument(float maturity, float strike, OptionType type);

} __attribute__((packed));

#endif