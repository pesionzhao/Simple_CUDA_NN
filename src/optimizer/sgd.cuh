#pragma once
#include"optim.cuh"

template<typename T>
class SGD : public Optimizer<T> {
public:
    float momentum;
    float weight_decay;
    SGD(std::vector<NNLayer<T>*> layers, float lr){
        this->lr = lr;
        for (NNLayer<T>* layer : layers) {
            for (std::shared_ptr<Tensor<T>> p : layer->params) {
                this->parameters.push_back(p);
            }
        }
    }
    void step() override{
        for (std::shared_ptr<Tensor<T>> p : this->parameters) {
            if (p->grad != nullptr) {
                dim3 block_size(32,32);
                dim3 grid_size((p->cols + block_size.x - 1)/ block_size.x, (p->rows + block_size.y - 1)/ block_size.y);
                addKernel<<<grid_size, block_size>>>(p->data_device.get(), p->grad->data_device.get(), this->lr, p->rows, p->cols);
            }
        }
    }
    void zero_grad() override{
        for (std::shared_ptr<Tensor<T>> p : this->parameters) {
            if (p->grad != nullptr) {
                p->grad->zeroInitDevice();
            }
        }
    }
};