#ifndef GEOMETRIC_ASIAN_H
#define GEOMETRIC_ASIAN_H

#include <asian/Asian.h>

class GeometricAsian : public Asian
{
  public:
    bool closedForm;

    GeometricAsian(bool closedForm, Asset asset, double interest, Instrument instrument,
                   bool useGpu, int pathNum, int observation);
    virtual Result calculate() override;
};

#endif