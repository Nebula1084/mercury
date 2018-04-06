#include <asian/Asian.h>
#include <vector>
#include <iostream>

void print(std::vector<std::vector<float>> matrix)
{
    for (auto &col : matrix)
    {
        for (auto &val : col)
        {
            std::cout << val << " ";
        }

        std::cout << std::endl;
    }
}
void cholesky()
{
    std::vector<std::vector<float>> corMatrix1 = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};

    Asian asin1(corMatrix1);
    auto res = asin1.cholesky();
    print(res);
    std::vector<std::vector<float>> corMatrix2 = {
        {1, 0.5},
        {0.5, 1}};
    Asian asin2(corMatrix2);
    res = asin2.cholesky();
    print(res);
}

int main()
{
    cholesky();
}
