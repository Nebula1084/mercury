#ifndef BASKET_EUROPEAN_H
#define BASKET_EUROPEAN_H

#include <comm/Protocol.h>
#include <option/Option.h>
#include <option/BlackScholes.h>
#include <simulate/MonteCarlo.h>

class BasketEuropean : public Option
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

    BasketEuropean(Protocol *buff);
    BasketEuropean(int basketSize, double interest, double repo, Instrument instrument,
                   Asset *asset, double *corMatrix, bool useGpu, int pathNum);
    BasketEuropean(const BasketEuropean &c);
    virtual ~BasketEuropean();

    double formulate();
    Result simulate(bool isGeo, bool control);
};

#endif