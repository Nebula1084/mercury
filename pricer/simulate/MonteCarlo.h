#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <option/Option.h>
#include <iostream>
#include <curand_kernel.h>
#include <cmath>

class Result
{
public:
  double arithPayoff;
  double geoPayoff;
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
  double optionValue(double value);

  Result simulateCPU(double *expectation, double *covMatrix);
  Result simulateGPU(double *expectation, double *covMatrix);
};

#endif