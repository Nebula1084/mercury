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

double BasketEuropean::formulate()
{
    double basketPrice = 1;
    for (int i = 0; i < basketSize; i++)
        basketPrice *= asset[i].price;
    basketPrice = std::pow(basketPrice, 1 / (double)basketSize);

    double sigma = 0;
    for (int i = 0; i < basketSize; i++)
        for (int j = 0; j < basketSize; j++)
            sigma += asset[i].volatility * asset[j].volatility * corMatrix[i * basketSize + j];
    sigma = std::sqrt(sigma) / (double)basketSize;

    double mu = 0;
    for (int i = 0; i < basketSize; i++)
        mu += asset[i].volatility * asset[i].volatility;
    mu = interest - 0.5 * mu / basketSize + 0.5 * sigma * sigma;

    Asset basketAsset(basketPrice, sigma, mu);
    BlackScholes formula(interest, 0, instrument, basketAsset);
    return formula.calculate();
}

Result BasketEuropean::simulate(bool isGeo, bool control)
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
                         instrument.strike, pathNum, 1, instrument.type, isGeo);
    if (control)
        simulator.setControlVariate(control, formulate());
    if (useGpu)
        return simulator.simulateGPU(expectation, covMatrix);
    else
        return simulator.simulateCPU(expectation, covMatrix);
}