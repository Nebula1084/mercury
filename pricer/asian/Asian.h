#ifndef ASIAN_H
#define ASIAN_H

#include <option/Option.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

class Asian
{
public:
  double drift;

  double price;
  double volatility;
  double interest;
  double maturity;
  double strike;
  OptionType type;
  int pathNum;
  int observation;

  Asian();
  Asian(double volatility, double interest, int observation);
};

#endif
