#pragma once
#include"../layers/linear_layer.cuh"
#include"../kernel/kernel.cuh"
template<typename T>
class Optimizer {
public:
    //只存可学习参数，也就是model的layer
    std::vector<std::shared_ptr<Tensor<T>>> parameters;
    float lr;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};