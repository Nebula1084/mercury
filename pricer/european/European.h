#ifndef EUROPEAN_H
#define EUROPEAN_H

#include <stdio.h>
#include <option/Option.h>
#include <option/BlackScholes.h>

class European
{
public:
  float interest;
  float repo;
  Instrument instrument;
  Asset asset;

  European(float interest, float repo, Instrument instrument, Asset asset);
  double calculate();

} __attribute__((packed));

#endif