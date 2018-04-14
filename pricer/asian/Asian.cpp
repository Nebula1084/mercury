#include <asian/Asian.h>

Asian::Asian()
{
}

Asian::Asian(int basketSize, double *corMatrix, double *volatility, double interest, int observation)
    : basketSize(basketSize), corMatrix(corMatrix), volatility(volatility), interest(interest), observation(observation)
{

}




static double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}