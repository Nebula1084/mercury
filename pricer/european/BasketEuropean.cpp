#include <european/BasketEuropean.h>

BasketEuropean::BasketEuropean(int basketSize, double interest, double repo, Instrument instrument, Asset *asset, double *corMatrix)
    : basketSize(basketSize), interest(interest), repo(repo), instrument(instrument)
{
    this->asset = new Asset[basketSize];
    for (int i = 0; i < basketSize; i++)
        this->asset[i] = asset[i];
    this->corMatrix = new double[basketSize * basketSize];
    for (int i = 0; i < basketSize * basketSize; i++)
        this->corMatrix[i] = corMatrix[i];
}

BasketEuropean::BasketEuropean(const BasketEuropean &c)
{
    this->basketSize = c.basketSize;
    this->interest = c.interest;
    this->repo = c.repo;
    this->instrument = c.instrument;

    this->asset = new Asset[basketSize];
    for (int i = 0; i < basketSize; i++)
        this->asset[i] = c.asset[i];
    this->corMatrix = new double[basketSize * basketSize];
    for (int i = 0; i < basketSize * basketSize; i++)
        this->corMatrix[i] = corMatrix[i];
}

BasketEuropean::~BasketEuropean()
{
    delete[] this->asset;
    delete[] this->corMatrix;
}