#include <european/GeometricEuropean.h>

GeometricEuropean::GeometricEuropean(bool closedForm, bool useGpu, int basketSize, double interest,
                                     Instrument instrument, Asset *asset, double *corMatrix, int pathNum)
    : closedForm(closedForm),
      BasketEuropean(basketSize, interest, 0, instrument, asset, corMatrix, useGpu, pathNum)
{
}

Result GeometricEuropean::calculate()
{
    Result result;
    if (closedForm)
    {
        result.mean = formulate();
        result.conf = -1;
    }
    else
        result = simulate(true, false);
    return result;
}
