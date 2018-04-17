#include <european/European.h>

European::European(Protocol *buff)
    : European(buff->interest, buff->repo, buff->instrument, buff->asset)
{
}

European::European(double interest, double repo, Instrument instrument, Asset asset)
    : interest(interest), repo(repo), instrument(instrument), asset(asset)
{
    this->asset.mean = -1;
}

Result European::calculate()
{
    Result result;
    BlackScholes formula(interest, repo, instrument, asset);
    result.mean = formula.calculate();
    result.conf = -1;
    return result;
}