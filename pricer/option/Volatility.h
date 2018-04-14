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
    Asset asset;     //need the underlying asset price for strike
    
  public:
    float calculate();
    Volatility();
    Volatility(float,float,Instrument,float,float);
  
} __attribute__((packed));

#endif