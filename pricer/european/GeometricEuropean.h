#ifndef GEOMETRIC_EUROPEAN_H
#define GEOMETRIC_EUROPEAN_H

#include <european/BasketEuropean.h>
#include <simulate/MonteCarlo.h>

class GeometricEuropean : public BasketEuropean
{
  public:
    bool closedForm;
    bool controlVariate;
    bool useGpu;
    int pathNum;

    GeometricEuropean(bool closedForm, bool controlVariate, bool useGpu, int basketSize, double interest,
                      Instrument instrument, Asset *asset, double *corMatrix, int pathNum);
    virtual double calculate() override;
    double formulate();
    double simulate();
};

#endif