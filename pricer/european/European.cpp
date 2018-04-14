#include <european/European.h>

European::European(double interest, double repo, Instrument instrument, Asset asset)
    : interest(interest), repo(repo), instrument(instrument), asset(asset)
{
}

double European::calculate()
{
    BlackScholes formula(interest, repo, instrument, asset);
    return formula.calculate();
}