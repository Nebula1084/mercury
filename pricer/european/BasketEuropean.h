#ifndef BASKET_EUROPEAN_H
#define BASKET_EUROPEAN_H

#include <option/Option.h>
#include <option/BlackScholes.h>

class BasketEuropean
{
  public:
    int basketSize;
    double interest;
    double repo;
    Instrument instrument;
    Asset *asset;
    double *corMatrix;

    BasketEuropean(int basketSize, double interest, double repo, Instrument instrument, Asset *asset, double *corMatrix);
    BasketEuropean(const BasketEuropean &c);
    virtual double calculate() = 0;
    virtual ~BasketEuropean();
};

#endif