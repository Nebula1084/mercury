#include <asian/Asian.h>
std::random_device rd;

std::mt19937 gen(rd());

Asian::Asian(std::vector<std::vector<float>> corMatrix)
    : corMatrix(corMatrix)
{
    this->basketSize = corMatrix.size();
    this->choMatrix = cholesky();
    this->simProdMatrix;
}

std::vector<std::vector<float>> Asian::cholesky()
{
    int n = this->corMatrix.size();
    std::vector<std::vector<float>> lMatrix;

    for (int i = 0; i < n; i++)
    {
        lMatrix.push_back(std::vector<float>(n));
        for (int j = 0; j <= i; j++)
        {
            lMatrix[i][j] = corMatrix[i][j];

            for (int k = 0; k < j; k++)
            {
                lMatrix[i][j] -= lMatrix[i][k] * lMatrix[j][k];
            }
            if (i == j)
            {
                lMatrix[i][j] = std::sqrt(lMatrix[i][j]);
            }
            else
            {
                lMatrix[i][j] = lMatrix[i][j] / lMatrix[j][j];
            }
        }
        for (int j = i + 1; j < n; j++)
        {
            lMatrix[i][j] = 0;
        }
    }

    return lMatrix;
}

std::vector<float> Asian::randNormal()
{
    std::vector<float> independNormals;
    std::vector<float> dependNormals;

    for (int i = 0; i < basketSize; i++)
    {
        std::normal_distribution<float> normal(0, 1);
        independNormals.push_back(normal(gen));
    }

    for (auto &comb : choMatrix)
    {
        float corNormal = 0;
        for (int i = 0; i < basketSize; i++)
        {
            corNormal += independNormals[i] * comb[i];
        }
        dependNormals.push_back(corNormal);
    }

    for (int i = 0; i < basketSize; i++)
    {
    }
    return dependNormals;
}