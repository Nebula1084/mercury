#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <option/Option.h>
#include <iostream>
#include <curand_kernel.h>
#include <cmath>

class Value
{
public:
  double expected;
  double confidence;
};

class MonteCarlo
{
public:
  int basketSize;
  double *price;
  double *volatility;
  double *corMatrix;

  double interest;
  double maturity;
  double strike;
  int pathNum;
  int observation;
  OptionType type;
  bool controlVariate;

  double *choMatrix;
  double *drift;
  MonteCarlo(int basketSize, double *corMatrix, double *volatility,
             double interest, int observation, OptionType type);

  double *cholesky();
  void randNormal(curandState *state, double *dependNormals);

  Value simulateCPU(double *expectation, double *covMatrix);
  Value simulateGPU(double *expectation, double *covMatrix);
};

#endif