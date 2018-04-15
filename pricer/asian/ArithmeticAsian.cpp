#include <asian/ArithmeticAsian.h>

ArithmeticAsian::ArithmeticAsian(bool controlVariate, Asset asset, double interest, Instrument instrument,
                                 bool useGpu, int pathNum, int observation)
    : Asian(asset, interest, instrument, useGpu, pathNum, observation),
      controlVariate(controlVariate)
{
}


Result ArithmeticAsian::calculate()
{
    return simulate(false, controlVariate);
}