#ifndef ASIAN_H
#define ASIAN_H

#include <iostream>

#include <comm/Protocol.h>
#include <option/Option.h>
#include <option/BlackScholes.h>
#include <simulate/MonteCarlo.h>

class Asian : public Option
{
public:
  Asset asset;
  double interest;
  Instrument instrument;
  int pathNum;
  int observation;
  bool useGpu;

  Asian(Protocol *buff);
  Asian(Asset asset, double interest, Instrument instrument, bool useGpu, int pathNum, int observation);

  double formulate();
  Result simulate(bool isGeo, bool control);
};

#endif
