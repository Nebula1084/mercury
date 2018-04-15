#include <asian/GeometricAsian.h>

GeometricAsian::GeometricAsian(bool closedForm, Asset asset, double interest, Instrument instrument,
                               bool useGpu, int pathNum, int observation)
    : Asian(asset, interest, instrument, useGpu, pathNum, observation),
      closedForm(closedForm)
{
}

Result GeometricAsian::calculate()
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