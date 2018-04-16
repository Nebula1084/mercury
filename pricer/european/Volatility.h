#ifndef _VOLATILITY_H
#define _VOLATILITY_H

#include <option/Option.h>
#include <european/European.h>

#include <stdio.h>
// #include <cmath>
#include <iostream>

class Volatility
{
private:
  double interest;
  double repo;
  Instrument instrument;
  double price;
  Asset asset; //need the underlying asset price for strike

public:
  double calculate();
  Volatility();
  Volatility(double, double, Instrument, double, double);

};

#endif