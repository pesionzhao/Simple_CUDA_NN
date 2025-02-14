#include "nn_utils/tensor.h"
#include "loss/mse.cuh"
#include "loss/crossentropy.cuh"
#include "layers/linear_layer.cuh"
#include "layers/relu_activation.cuh"
// #include "layers/softmax.cuh"
#include "optimizer/sgd.cuh"
#include "optimizer/adam.cuh"
#include "module/module.cuh"
#include "mnist/mnist.h"
int main(){
    manual_seed(&rng, 1142);
    const std::string path = "/workspaces/CUDNN/src/dataset/mnist_test_10.csv";
    Mnist mnist(path);
    Loss<float>* mse = new CE<float>();
    Network<float> net;
    std::shared_ptr<Tensor<float>> Y = std::make_shared<Tensor<float>>();
    std::shared_ptr<Tensor<float>> m = std::make_shared<Tensor<float>>();
    std::shared_ptr<Tensor<float>> label = std::make_shared<Tensor<float>>();
    // Optimizer<float>* optim = new SGD<float>(net.layers, 0.001);
    Optimizer<float>* optim = new Adam<float>(net.layers, 0.001);
    int epoch = 100;
    for(int i = 0; i<epoch; i++)
    {
        std::cout<<"epoch ["<<i+1<<"/"<<epoch<<"], ";
        for(int batch = 0; batch<mnist.num_batch; batch++){
            m = mnist.getItem(batch);
            label = mnist.getLabel(batch);
            // for(int t = 0; t<10; t++) std::cout<<label->data_host.get()[t]<< "  ";
            Y = net.forward(m);
            mse->cost(Y, label);
            mse->backwardPass();
            // net.backward(Y, label); //计算梯度
            optim->step();//更新权重
            optim->zero_grad();
        }
    }
}