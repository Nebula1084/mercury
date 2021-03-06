#ifndef GEOMETRIC_EUROPEAN_H
#define GEOMETRIC_EUROPEAN_H

#include <european/BasketEuropean.h>

class GeometricEuropean : public BasketEuropean
{
  public:
    bool closedForm;

    GeometricEuropean(Protocol *buff);
    GeometricEuropean(bool closedForm, bool useGpu, int basketSize, double interest,
                      Instrument instrument, Asset *asset, double *corMatrix, int pathNum);
    virtual Result calculate() override;
};

#endif