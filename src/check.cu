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
    // std::shared_ptr<Tensor<float>> m = std::make_shared<Tensor<float>>(1,5);
    // m->randomInitHost();
    // std::cout << *m <<std::endl; 
    // std::shared_ptr<Tensor<float>> label = std::make_shared<Tensor<float>>(10,1);
    // std::cout<< "Tensor (" << m->rows << ", " << m->cols << "):" << std::endl;

    /*================自定义初始化================*/
    // for(int i = 0; i<m->rows; i++){
    //     m->data_host.get()[i] = (float)i/10;
    //     label->data_host.get()[i] = i==4?1:0;
    // }
    // m->copyHostToDevice();
    // label->copyHostToDevice();
    // std::cout<<*label<<std::endl;
    // m.allocate();
    // std::cout<<"allocate done"<<std::endl;
    /*================end================*/

    // m->randomInitDevice(31);
    // label->zeroInitHost();
    // label->data_host.get()[4] = 1;
    // label->copyHostToDevice();
    // std::cout<<"init done"<<std::endl;
    // std::cout<<*m<<std::endl;
    /*================测试CE================*/
    // Loss<float>* mse = new CE<float>();
    // mse->cost(m, label);
    // std::shared_ptr<Tensor<float>> dL =  mse->dCost();
    // dL->copyDeviceToHost();
    // std::cout<<(*dL)[0]<<std::endl;
    /*================end================*/
    std::shared_ptr<Tensor<float>> m = std::make_shared<Tensor<float>>(4,4);
    m->randomInitDevice();
    m->copyDeviceToHost();
    std::cout<<*m;
    std::shared_ptr<Tensor<float>> res = sum_(m.get(), 1);
    res->copyDeviceToHost();
    std::cout<<*res;
    return 0;
}