#include "Matrix.h"
#include"../layers/linear_layer.cuh"
int main(){
    Matrix<float> m(1024,1);
    Matrix<float> dL(256,1);
    std::cout<< "Matrix (" << m.rows << ", " << m.cols << "):" << std::endl;
    // m.allocate();
    // std::cout<<"allocate done"<<std::endl;
    m.randomInitDevice();
    std::cout<<"init done"<<std::endl;
    m.copyDeviceToHost();
    // std::cout<<m<<std::endl;
    LinearLayer<float> layer(1024, 256);
    Matrix<float> Y = layer.forward(m);
    Matrix<float> dm = layer.backward(dL);
    Y.copyDeviceToHost();
    // std::cout<<Y<<std::endl;
    m.save_to_file("A.bin");
    layer.save_data();
}