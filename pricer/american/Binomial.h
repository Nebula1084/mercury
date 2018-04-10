#ifndef BINOMIAL_H
#define BINOMIAL_H

#include <american/American.h>

#define MAX_OPTIONS 1024
#define NUM_STEPS 2048

void binomialOptionsGPU(float *callValue, American *americans, int num);

#endif