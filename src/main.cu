#include "nn_utils/Matrix.h"
#include "loss/mse.cuh"
#include "loss/crossentropy.cuh"
#include "layers/linear_layer.cuh"
#include "layers/softmax.cuh"
#include "optimizer/sgd.cuh"
#include "optimizer/adam.cuh"
#include "module/module.cuh"
int main(){
    std::shared_ptr<Matrix<float>> m = std::make_shared<Matrix<float>>(1024,1);
    std::shared_ptr<Matrix<float>> label = std::make_shared<Matrix<float>>(128,1);
    std::cout<< "Matrix (" << m->rows << ", " << m->cols << "):" << std::endl;

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

    m->randomInitDevice(31);
    label->randomInitDevice(12);
    std::cout<<"init done"<<std::endl;
    // std::cout<<*m<<std::endl;
    /*================测试CE================*/
    // Loss<float>* mse = new CE<float>();
    // mse->cost(m, label);
    // std::shared_ptr<Matrix<float>> dL =  mse->dCost();
    // dL->copyDeviceToHost();
    // std::cout<<(*dL)[0]<<std::endl;
    /*================end================*/
    
    Loss<float>* mse = new MSE<float>();
    Network<float> net;
    net.addLayer(new LinearLayer<float>(1024, 128));
    net.addLayer(new LinearLayer<float>(128, 128));
    // net.addLayer(new LinearLayer<float>(128, 128));

    std::shared_ptr<Matrix<float>> Y = std::make_shared<Matrix<float>>();
    // Optimizer<float>* optim = new SGD<float>(net.layers, 0.001);
    Optimizer<float>* optim = new Adam<float>(net.layers, 0.001);
    for(int i = 0; i<10; i++)
    {
        std::cout<<"epoch "<<i<<"  ";
        Y = net.forward(m);
        mse->cost(Y, label);
        mse->backwardPass();
        // net.backward(Y, label); //计算梯度
        optim->step();//更新权重
        optim->zero_grad();

    }
}