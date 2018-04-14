#include <option/Norm.h>

double normCdf(double d)
{
	const double A0 = 0.2316419;
	const double A1 = 0.31938153;
	const double A2 = -0.356563782;
	const double A3 = 1.781477937;
	const double A4 = -1.821255978;
	const double A5 = 1.330274429;

	double RSQRT2PI = std::sqrt(2 * PI);
	double cnd, K, L;
	K = 1.0 / (1.0 + A0 * std::fabs(d));
	L = std::pow(std::fabs(d), 2);

	cnd = 1.0 - 1.0 / std::sqrt(2 * PI) * std::exp(-0.5 * L) * (A1 * K + A2 * K * K + A3 * std::pow(K, 3) + A4 * std::pow(K, 4) + A5 * std::pow(K, 5));

	if (d < 0)
		cnd = 1.0 - cnd;

	return cnd;
}

double normPdf(double d)
{
	double pdf = (1.0 / std::sqrt(2.0 * PI)) * std::exp(-0.5 * d * d);
	return pdf;
}