#ifndef EUROPEAN_H
#define EUROPEAN_H

#include <option/Option.h>
#include <option/BlackScholes.h>

class European
{
public:
  double interest;
  double repo;
  Instrument instrument;
  Asset asset;

  European(double interest, double repo, Instrument instrument, Asset asset);
  Result calculate();
};

#endif