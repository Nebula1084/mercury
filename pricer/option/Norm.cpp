#include <option/Norm.h>

float norm_cdf(float d)
{
	const float A0 = 0.2316419;
	const float A1 = 0.31938153;
	const float A2 = -0.356563782;
	const float A3 = 1.781477937;
	const float A4 = -1.821255978;
	const float A5 = 1.330274429;

	float RSQRT2PI = sqrt(2*PI);
	float cnd,K,L;
	K = 1.0/(1.0+A0*fabs(d));
	L = pow(fabs(d),2);

	cnd = 1.0-1.0/sqrt(2*PI)*exp(-0.5*L)*(A1*K+A2*K*K+A3*pow(K,3)+A4*pow(K,4)+A5*pow(K,5));
	
    if (d<0)
		cnd = 1.0-cnd;

	return cnd;
}

float norm_pdf(float d)
{
    float pdf = (1.0/sqrt(2.0*PI))*exp(-0.5*d*d);
    return pdf; 
}