#ifndef EUROPEAN_H
#define EUROPEAN_H

#include <comm/Protocol.h>
#include <option/Option.h>
#include <option/BlackScholes.h>

class European : public Option
{
public:
  double interest;
  double repo;
  Instrument instrument;
  Asset asset;

  European(Protocol *buff);
  European(double interest, double repo, Instrument instrument, Asset asset);
  virtual Result calculate() override;
};

#endif