#ifndef ASIAN_H
#define ASIAN_H

#include <vector>
#include <cmath>
#include <iostream>

class Asian
{
private:
  std::vector<std::vector<float>> corMatrix;

public:
  Asian(std::vector<std::vector<float>> corMatrix);
  std::vector<std::vector<float>> cholesky();
};

#endif
