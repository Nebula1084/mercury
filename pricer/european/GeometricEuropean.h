#ifndef GEOMETRIC_EUROPEAN_H
#define GEOMETRIC_EUROPEAN_H

#include <european/BasketEuropean.h>

class GeometricEuropean : public BasketEuropean
{
    GeometricEuropean(int basketSize, double interest, double repo, Instrument instrument, Asset *asset, double *corMatrix);
    virtual double calculate() override;
};

#endif