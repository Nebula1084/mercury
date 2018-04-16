#include <option/Option.h>

Asset::Asset()
{
}

Asset::Asset(double price, double volatility)
    : price(price), volatility(volatility), mean(-1)
{
}

Asset::Asset(double price, double volatility, double mean)
    : price(price), volatility(volatility), mean(mean)
{
}

void Asset::setVolatility(double volatility)
{
    this->volatility = volatility;
}

Instrument::Instrument()
{
}

Instrument::Instrument(double maturity, double strike, OptionType type)
{
    this->maturity = maturity;
    this->strike = strike;
    this->type = type;
}

std::ostream &operator<<(std::ostream &out, const Asset &asset)
{
    out << "Price:" << asset.price << " Volatility :" << asset.volatility;
    return out;
}

std::ostream &operator<<(std::ostream &out, const Instrument &instrument)
{
    out << "Strike:" << instrument.strike << " Maturity :" << instrument.maturity;
    out << " Type:";
    if (instrument.type == CALL)
        out << "CALL";
    else if (instrument.type == PUT)
        out << "PUT";
    else
        out << "Invalid";
    return out;
}

std::ostream &operator<<(std::ostream &out, const Result &result)
{
    out << "Mean:" << result.mean << " Conf:" << result.conf;
    return out;
}