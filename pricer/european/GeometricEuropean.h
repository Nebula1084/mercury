#ifndef GEOMETRIC_EUROPEAN_H
#define GEOMETRIC_EUROPEAN_H

#include <european/BasketEuropean.h>
#include <simulate/MonteCarlo.h>

class GeometricEuropean : public BasketEuropean
{
    bool closedForm;
    bool controlVariate;
    bool useGpu;

    GeometricEuropean(bool closedForm, bool controlVariate, bool useGpu, int basketSize, double interest, double repo,
                      Instrument instrument, Asset *asset, double *corMatrix);
    virtual double calculate() override;
    double formulate();
    double simulate();
};

#endif