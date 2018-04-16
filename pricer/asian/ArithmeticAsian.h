#ifndef ARITHMETIC_ASIAN_H
#define ARITHMETIC_ASIAN_H

#include <asian/Asian.h>

class ArithmeticAsian : public Asian
{
  public:
    bool controlVariate;

    ArithmeticAsian(Protocol *buff);
    ArithmeticAsian(bool controlVariate, Asset asset, double interest, Instrument instrument,
                    bool useGpu, int pathNum, int observation);
    virtual Result calculate() override;
};

#endif