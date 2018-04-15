#ifndef BASKET_EUROPEAN_H
#define BASKET_EUROPEAN_H

#include <option/Option.h>
#include <option/BlackScholes.h>
#include <simulate/MonteCarlo.h>

class BasketEuropean
{
  public:
    int basketSize;
    double interest;
    double repo;
    Instrument instrument;
    Asset *asset;
    double *corMatrix;
    bool useGpu;
    int pathNum;

    BasketEuropean(int basketSize, double interest, double repo, Instrument instrument,
                   Asset *asset, double *corMatrix, bool useGpu, int pathNum);
    BasketEuropean(const BasketEuropean &c);
    virtual ~BasketEuropean();

    virtual double calculate() = 0;
    double formulate();
    Result simulate(bool isGeo, bool control);
};

#endif