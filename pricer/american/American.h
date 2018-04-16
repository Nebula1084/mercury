#ifndef AMERICAN_H
#define AMERICAN_H

#include <comm/Protocol.h>
#include <option/Option.h>
#include <option/Norm.h>

class American : public Option
{
  public:
    bool useGpu;

    double interest;
    Asset asset;
    Instrument instrument;
    int step;

    American(Protocol *buff);

    American(bool useGpu, double interest, Asset asset, Instrument instrument, int step);
    double optionValue(int i, int j);
    double binomialCPU();
    double binomialGPU();

    virtual Result calculate() override;

    double dt, vDt, rDt, If, Df, u, d;
    double pu, pd, puByDf, pdByDf;

  private:
    void preprocess();
};

#endif