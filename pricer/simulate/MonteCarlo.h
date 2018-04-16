#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <option/Option.h>
#include <iostream>
#include <curand_kernel.h>
#include <cmath>

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
  bool isGeo;

  double geoExp;

  double *choMatrix;
  double *drift;
  MonteCarlo(int basketSize, double *price, double *corMatrix, double *volatility,
             double interest, double maturity, double strike, int pathNum,
             int observation, OptionType type, bool isGeo);

  void setControlVariate(bool control, double geoExp);

  double confidence(double std);

  double *cholesky();
  void randNormal(curandState *state, double *dependNormals);
  double optionValue(double value);

  void statistic(double *values, double &mean, double &std);
  void statisticGPU(MonteCarlo *plan, double *value, double &mean, double &std);

  double covariance(double *arith, double *geo, double arithMean, double geoMean);
  double covarianceGPU(MonteCarlo *plan, double *arith, double *geo, double arithMean, double geoMean);

  void variationReduce(double *dst, double *arithPayoff, double *geoPayoff, double theta);

  Result simulateCPU(double *expectation, double *covMatrix);
  Result simulateGPU(double *expectation, double *covMatrix);
};

#endif