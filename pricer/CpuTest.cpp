#include <iostream>
#include <option/European.h>
#include <option/Volatility.h>

using namespace std;

int main()
{
    Instrument instrument1(0.0219,1.8,CALL);
    Volatility volatility1(0.04,0.2,instrument1,0.1547,1.9595);

    Instrument instrument2(0.0219,2.1,2);
    Volatility volatility2(0.04,0.2,instrument2,0.143,1.9595);

    Instrument instrument3(0.0219,2.1,2);
    Volatility volatility3(0.04,0.2,instrument3,0.1599,1.9595);

    Instrument instrument4(0.0219,2.5,1);
    Volatility volatility4(0.04,0.2,instrument4,0.0001,1.9595);

    Instrument instrument5(0.0219,2.5,2);
    Volatility volatility5(0.04,0.2,instrument5,0.4828,1.9595);

    Instrument instrument6(0.0219,2.5,1);
    Volatility volatility6(0.04,0.2,instrument6,0.0004,1.9595);

    Instrument instrument7(0.0219,2.5,1);
    Volatility volatility7(0.04,0.2,instrument7,0.6251,1.9595);

    float sigma1 = volatility1.calculate();
    cout<<sigma1<<endl;

    float sigma2 = volatility2.calculate();
    cout<<sigma2<<endl;

    float sigma3 = volatility3.calculate();
    cout<<sigma3<<endl;
    
    float sigma4 = volatility4.calculate();
    cout<<sigma4<<endl;

    float sigma5 = volatility5.calculate();
    cout<<sigma5<<endl;

    float sigma6 = volatility6.calculate();
    cout<<sigma6<<endl;

    float sigma7 = volatility7.calculate();
    cout<<sigma7<<endl;

    return 0;
}