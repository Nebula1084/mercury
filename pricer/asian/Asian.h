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
  double *corMatrix;
  double *choMatrix;
  double *drift;

  int basketSize;

  double *price;
  double *volatility;
  double interest;
  double maturity;
  double strike;
  OptionType type;
  int pathNum;
  int observation;

  Asian();
  Asian(int basketSize, double *corMatrix, double *volatility, double interest, int observation);
};

#endif
