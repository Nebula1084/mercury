#include <iostream>
#include <option/European.h>
#include <option/Volatility.h>

using namespace std;

int main()
{
    Asset asset1(100.0,0.2);
    Instrument instrument1(0.5,100.0,2);
    European european1(0.01,0,instrument1,asset1);

    Asset asset2(100.0,0.2);
    Instrument instrument2(0.5,120.0,2);
    European european2(0.01,0,instrument2,asset2);

    Asset asset3(100.0,0.2);
    Instrument instrument3(1.0,100.0,2);
    European european3(0.01,0,instrument3,asset3);

    Asset asset4(100.0,0.3);
    Instrument instrument4(0.5,100.0,2);
    European european4(0.01,0,instrument4,asset4);

    Asset asset5(100.0,0.2);
    Instrument instrument5(0.5,100.0,2);
    European european5(0.02,0,instrument5,asset5);
    
    float v1 = european1.calculate();
    float v2 = european2.calculate();
    float v3 = european3.calculate();
    float v4 = european4.calculate();
    float v5 = european5.calculate();

    cout<<"Call Option:"<<endl;
    cout<<v1<<endl;
    cout<<v2<<endl;
    cout<<v3<<endl;
    cout<<v4<<endl;
    cout<<v5<<endl;

    return 0;
}