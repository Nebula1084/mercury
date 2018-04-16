#ifndef OPTION_H
#define OPTION_H

#include <iostream>

typedef char OptionType;

const OptionType CALL = 1;
const OptionType PUT = 2;

struct Asset
{
  public:
    double price;
    double mean;
    double volatility;

    Asset();
    Asset(double price, double volatility);
    Asset(double price, double volatility, double mean);

    void setVolatility(double volatility);

    friend std::ostream &operator<<(std::ostream &out, const Asset &asset);
} __attribute__((packed));

struct Instrument
{
  public:
    double maturity;
    double strike;
    OptionType type;

    Instrument();
    Instrument(double maturity, double strike, OptionType type);

    friend std::ostream &operator<<(std::ostream &out, const Instrument &instrument);
} __attribute__((packed));

struct Result
{
  public:
    double mean;
    double conf;
    friend std::ostream &operator<<(std::ostream &out, const Result &result);
} __attribute__((packed));

class Option
{
  public:
    virtual Result calculate() = 0;
};

#endif