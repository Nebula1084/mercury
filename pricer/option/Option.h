#ifndef OPTION_H
#define OPTION_H

typedef char OptionType;

const OptionType CALL = 1;
const OptionType PUT = 2;

typedef char Operation;

const Operation EUROPEAN = 1;
const Operation VOLATILITY = 2;
const Operation AMERICAN = 3;

class Asset
{
private:
  float price;
  float volatility;
} __attribute__((packed));

class Instrument
{
private:
  float maturity;
  float strike;
  OptionType type;
} __attribute__((packed));
#endif