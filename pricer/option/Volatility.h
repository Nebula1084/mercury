#ifndef _VOLATILITY_H
#define _VOLATILITY_H

#include <option/Option.h>
#include <stdio.h>

class Volatility
{
  private:
    float interest;
    float repo;
    Instrument instrument;
    float price;

  public:
    float calculate();
} __attribute__((packed));

#endif