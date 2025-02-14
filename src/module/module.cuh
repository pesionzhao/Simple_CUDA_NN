#pragma once
#include "../layers/linear_layer.cuh"
#include "../layers/linear_layer_official.cuh"
#include "../layers/relu_activation.cuh"
#include "../loss/mse.cuh"
#include "../nn_utils/rand.h"
template <typename T>
class Network
{
private:
    std::shared_ptr<Tensor<T>> Y;

public:
    std::vector<NNLayer<T>*> layers;
    Network(){
        layers.push_back(new LinearLayer_torch<T>(28 * 28, 64));
        layers.push_back(new Relu<T>());
        layers.push_back(new LinearLayer_torch<T>(64, 10));
        layers.push_back(new Relu<T>());
    }
    void addLayer(NNLayer<T> *layer){
        layers.push_back(layer);
    }
    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> X){
        // 按照vector遍历
        std::shared_ptr<Tensor<T>> A = X;
        for (auto layer : layers)
        {
            A = layer->forward(A);
        }
        Y = A;
        return Y;
    }
};