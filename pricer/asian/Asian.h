#ifndef ASIAN_H
#define ASIAN_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>

class Asian
{
private:
  std::vector<std::vector<float>> corMatrix;
  std::vector<std::vector<float>> choMatrix;

  std::vector<std::vector<float>> simProdMatrix;
  std::vector<float> simSumVector;
  std::vector<float> simSqrSumVector;

  int basketSize;

public:
  Asian(std::vector<std::vector<float>> corMatrix);
  std::vector<std::vector<float>> cholesky();
  std::vector<float> randNormal();
};

#endif
