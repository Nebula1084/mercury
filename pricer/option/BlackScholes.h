#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include <option/Option.h>
#include <option/Norm.h>

class BlackScholes
{
public:
  double interest;
  double repo;
  Instrument instrument;
  Asset asset;

  BlackScholes(double interest, double repo, Instrument instrument, Asset asset);
  double calculate();
  double vega();

};

#endif