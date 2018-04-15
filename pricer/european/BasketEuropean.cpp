#include <european/BasketEuropean.h>

BasketEuropean::BasketEuropean(int basketSize, double interest, double repo, Instrument instrument,
                               Asset *asset, double *corMatrix, bool useGpu, int pathNum)
    : basketSize(basketSize), interest(interest), repo(repo), instrument(instrument),
      useGpu(useGpu), pathNum(pathNum)
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

Result BasketEuropean::simulate()
{
    double volatility[basketSize];
    double price[basketSize];
    double covMatrix[basketSize * basketSize];
    double expectation[basketSize];

    for (int i = 0; i < basketSize; i++)
    {
        price[i] = asset[i].price;
        volatility[i] = asset[i].volatility;
    }
    MonteCarlo simulator(basketSize, price, corMatrix, volatility, interest, instrument.maturity,
                         instrument.strike, pathNum, 1, instrument.type);
    if (useGpu)
        return simulator.simulateGPU(expectation, covMatrix);
    else
        return simulator.simulateCPU(expectation, covMatrix);
}