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

std::ostream &operator<<(std::ostream &out, const Result result)
{
    out << "Mean:" << result.mean << " Conf:" << result.conf;
    return out;
}