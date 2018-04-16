#include <comm/Protocol.h>
#include <european/European.h>
#include <european/Volatility.h>
#include <european/GeometricEuropean.h>
#include <european/ArithmeticEuropean.h>
#include <american/American.h>
#include <asian/GeometricAsian.h>
#include <asian/ArithmeticAsian.h>

Option *Protocol::parse(Protocol *buff)
{
    Option *task;
    switch (buff->operation)
    {
    case EUROPEAN:
        task = new European(buff);
        break;
    case VOLATILITY:
        task = new Volatility(buff);
        break;
    case GEOMETRIC_EUROPEAN:
        task = new GeometricEuropean(buff);
        break;
    case ARITHMETIC_EUROPEAN:
        task = new ArithmeticEuropean(buff);
        break;
    case AMERICAN:
        task = new American(buff);
        break;
    case GEOMETRIC_ASIAN:
        task = new GeometricAsian(buff);
        break;
    case ARITHMETIC_ASIAN:
        task = new ArithmeticAsian(buff);
        break;
    }

    return task;
}