#include <european/ArithmeticEuropean.h>

ArithmeticEuropean::ArithmeticEuropean(bool controlVariate, bool useGpu, int basketSize, double interest,
                                       Instrument instrument, Asset *asset, double *corMatrix, int pathNum)
    : controlVariate(controlVariate),
      BasketEuropean(basketSize, interest, 0, instrument, asset, corMatrix, useGpu, pathNum)
{
}

double ArithmeticEuropean::calculate()
{
    return simulate().arithPayoff;
}