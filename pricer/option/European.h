#ifndef EUROPEAN_H
#define EUROPEAN_H

#include <option/Option.h>
#include <stdio.h>

class European
{
private:
  //friend class Volatility;
  float interest;
  float repo;
  Instrument instrument;
  Asset asset;
  
public:
  float calculate();
  European();
  European(float r, float repo, Instrument instrument, Asset asset);

} __attribute__((packed));

#endif