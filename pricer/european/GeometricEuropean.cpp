#include <european/GeometricEuropean.h>

GeometricEuropean::GeometricEuropean(bool closedForm, bool useGpu, int basketSize, double interest,
                                     Instrument instrument, Asset *asset, double *corMatrix, int pathNum)
    : closedForm(closedForm),
      BasketEuropean(basketSize, interest, 0, instrument, asset, corMatrix, useGpu, pathNum)
{
}

double GeometricEuropean::calculate()
{
    if (closedForm)
        return formulate();
    else
        return simulate(true, false).geoPayoff;
}
