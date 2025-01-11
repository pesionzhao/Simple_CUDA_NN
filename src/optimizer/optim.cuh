#pragma once
#include"../layers/linear_layer.cuh"
#include"../kernel/kernel.cuh"
template<typename T>
class Optimizer {
public:
    std::vector<std::shared_ptr<Matrix<T>>> parameters;
    float lr;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};