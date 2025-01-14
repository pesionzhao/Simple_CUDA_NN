//similar to  nn.module
#pragma once
#include"../layers/linear_layer.cuh"
#include"../loss/mse.cuh"
template<typename T>
class Network{
private:
	std::shared_ptr<Matrix<T>> Y;
public:
	std::vector<NNLayer<T>*> layers;
    Network(){
    }
    void addLayer(NNLayer<T>* layer){
        layers.push_back(layer);
    }
    std::shared_ptr<Matrix<T>> forward(std::shared_ptr<Matrix<T>> X){
        std::shared_ptr<Matrix<T>> A = X;
        for(auto layer : layers){
            A = layer->forward(A);
        }
        Y = A;
        return Y;
    }
    // void backward(std::shared_ptr<Matrix<T>> prediction, std::shared_ptr<Matrix<T>> target){
    //     float loss = func->cost(prediction, target);
    //     std::shared_ptr<Matrix<T>> dY = func->dCost();
    //     std::cout<< "cost: "<< loss << std::endl;
    //     for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
    //         dY = (*it)->backward(dY);
    //         // std::cout<< "layer done, dy shape = ["<< dY.rows<<", "<<dY.cols<<"]"<<std::endl;
    //     }
    //     cudaDeviceSynchronize();//确保之前的CUDA操作都已完成
    // }
};