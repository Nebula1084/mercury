#ifndef AMERICAN_H
#define AMERICAN_H

#include <math.h>

class American
{
  public:
    float S;
    float X;
    float T;
    float R;
    float V;

    const static int NUM_STEPS;

    static double CND(double d);
    static double expiryCallValue(double S, double X, double vDt, int i);
    float BlackScholesCall();
    float binomialOptionsCPU();
};

#endif