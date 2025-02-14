//用于初步检查kernel计算的正误
#include "nn_utils/tensor.h"
#include "loss/mse.cuh"
#include "loss/crossentropy.cuh"
#include "layers/linear_layer.cuh"
#include "layers/relu_activation.cuh"
// #include "layers/softmax.cuh"
#include "optimizer/sgd.cuh"
#include "optimizer/adam.cuh"
#include "module/module.cuh"
#include "minist/minist.h"
int main(){
    manual_seed(&rng, 1142);
    std::shared_ptr<Tensor<float>> m = std::make_shared<Tensor<float>>(4,4);
    m->randomInitDevice();
    m->copyDeviceToHost();
    std::cout<<*m;
    std::shared_ptr<Tensor<float>> res = sum_(m.get(), 1);
    res->copyDeviceToHost();
    std::cout<<*res;
    return 0;
}