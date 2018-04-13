#ifndef ASIAN_H
#define ASIAN_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <curand_kernel.h>

class Asian
{
public:
  double *corMatrix;
  double *choMatrix;

  int basketSize;

  double S;
  double X;
  double T;
  double R;
  double V;
  int pathNum;

  class Value
  {
  public:
    double expected;
    double confidence;
  };

  Asian();
  Asian(int basketSize, double *corMatrix);

  double *cholesky();
  double *randNormal(curandState *state);
  Value monteCarloCPU(int pathN);
};

#endif
