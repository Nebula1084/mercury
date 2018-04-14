#include <european/European.h>

European::European(float interest, float repo, Instrument instrument, Asset asset)
    : interest(interest), repo(repo), instrument(instrument), asset(asset)
{
}

double European::calculate()
{
    BlackScholes formula(interest, repo, instrument, asset);
    return formula.calculate();
}