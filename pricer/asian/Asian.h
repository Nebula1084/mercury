#ifndef ASIAN_H
#define ASIAN_H

#include <iostream>

#include <option/Option.h>
#include <option/BlackScholes.h>
#include <simulate/MonteCarlo.h>

class Asian
{
public:
  Asset asset;
  double interest;
  Instrument instrument;
  int pathNum;
  int observation;
  bool useGpu;

  Asian(Asset asset, double interest, Instrument instrument, bool useGpu, int pathNum, int observation);

  virtual Result calculate() = 0;
  double formulate();
  Result simulate(bool isGeo, bool control);
};

#endif
