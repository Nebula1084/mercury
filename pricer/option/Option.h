#ifndef OPTION_H
#define OPTION_H

#include <iostream>

typedef char OptionType;

const OptionType CALL = 1;
const OptionType PUT = 2;

typedef char Operation;

const Operation EUROPEAN = 1;
const Operation VOLATILITY = 2;
const Operation AMERICAN = 3;

class Asset
{
public:
  double price;
  double mean;
  double volatility;

  Asset();
  Asset(double price, double volatility, double mean);

  void setVolatility(double volatility);
} __attribute__((packed));

class Instrument
{
public:
  double maturity;
  double strike;
  OptionType type;

  Instrument();
  Instrument(double maturity, double strike, OptionType type);

} __attribute__((packed));

class Result
{
public:
  double mean;
  double conf;
  friend std::ostream &operator<<(std::ostream &out, const Result result);
};

#endif