#include <american/American.h>

const int American::NUM_STEPS = 2048;

double American::CND(double d)
{
    const double A1 = 0.31938153;
    const double A2 = -0.356563782;
    const double A3 = 1.781477937;
    const double A4 = -1.821255978;
    const double A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double cnd = RSQRT2PI * exp(-0.5 * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

double American::expiryCallValue(double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}

float American::BlackScholesCall()
{
    double sqrtT = sqrt(T);

    double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double d2 = d1 - V * sqrtT;

    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(-R * T);

    float callResult = (float)(S * CNDD1 - X * expRT * CNDD2);
    return callResult;
}

float American::binomialOptionsCPU()
{

    static double Call[NUM_STEPS + 1];

    const double dt = T / (double)NUM_STEPS;
    const double vDt = V * sqrt(dt);
    const double rDt = R * dt;

    const double If = exp(rDt);
    const double Df = exp(-rDt);

    const double u = exp(vDt);
    const double d = exp(-vDt);
    const double pu = (If - d) / (u - d);
    const double pd = 1.0 - pu;
    const double puByDf = pu * Df;
    const double pdByDf = pd * Df;

    for (int i = 0; i <= NUM_STEPS; i++)
        Call[i] = expiryCallValue(S, X, vDt, i);

    for (int i = NUM_STEPS; i > 0; i--)
        for (int j = 0; j <= i - 1; j++)
            Call[j] = puByDf * Call[j + 1] + pdByDf * Call[j];

    return (float)Call[0];
}
