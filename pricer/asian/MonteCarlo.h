#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <asian/Asian.h>

Asian::Value monteCarloGPU(Asian *asian, double *expectation, double *covMatrix);

#endif