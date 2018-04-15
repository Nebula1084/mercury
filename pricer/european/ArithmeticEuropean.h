#ifndef ARITHMETIC_EUROPEAN_H
#define ARITHMETIC_EUROPEAN_H

#include <european/BasketEuropean.h>
#include <simulate/MonteCarlo.h>

class ArithmeticEuropean : public BasketEuropean
{
  public:
    bool controlVariate;
    ArithmeticEuropean(bool controlVariate, bool useGpu, int basketSize, double interest,
                       Instrument instrument, Asset *asset, double *corMatrix, int pathNum);
    virtual Result calculate() override;
};
#endif