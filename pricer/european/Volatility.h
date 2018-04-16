#ifndef _VOLATILITY_H
#define _VOLATILITY_H

#include <comm/Protocol.h>
#include <option/Option.h>
#include <european/European.h>

class Volatility : public Option
{
  private:
    double interest;
    double repo;
    Instrument instrument;
    double premium;
    Asset asset; //need the underlying asset price for strike

  public:
    Volatility(Protocol *buff);
    Volatility(double interest, double repo, Instrument instrument, double premium, double S);
    virtual Result calculate() override;
};

#endif