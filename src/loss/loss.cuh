#pragma once

#include <string>
#include <vector>
#include "../nn_utils/Matrix.h"

template<typename T>
class Loss{
protected:
    std::string name;
    std::shared_ptr<Matrix<T>> predictions;
    std::shared_ptr<Matrix<T>> target;
public:
    virtual float cost(std::shared_ptr<Matrix<T>> predictions, std::shared_ptr<Matrix<T>> target) = 0;
    virtual std::shared_ptr<Matrix<T>> dCost() = 0;
    const std::string getName() const { return this->name; };
};