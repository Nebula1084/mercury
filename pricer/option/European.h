#ifndef EUROPEAN_H
#define EUROPEAN_H

#include <option/Option.h>
#include <stdio.h>

class European
{
private:
  float interest;
  float repo;
  Instrument instrument;
  Asset asset;

public:
  float calculate();

} __attribute__((packed));

#endif