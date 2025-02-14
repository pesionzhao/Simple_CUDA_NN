#pragma once
#include "nn_layer.h"
#include "../kernel/kernel.cuh"
//矩阵乘法 Y= W@A+b
//W: [M, K]
//A: [K, N]
//b: [M, N]
//Y: [M, N]

//线性层: output = input@W^T+b
template<typename T>
class LinearLayer_torch : public NNLayer<T> {
public:
    std::shared_ptr<Tensor<T>> W;
    std::shared_ptr<Tensor<T>> b;
    std::shared_ptr<Tensor<T>> output;
    std::shared_ptr<Tensor<T>> input;
    int input_feature;
    int output_feature;
public:
    LinearLayer_torch(int input_size, int output_size){
        this->input_feature = input_size;
        this->output_feature = output_size;
        this->name = "Linear";
        W = std::make_shared<Tensor<T>>(output_size, input_size, this->train);
        b = std::make_shared<Tensor<T>>(1, output_size, this->train);
        init();
        this->params = {W, b};
    }
    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input){
        this->input = input;
        // output = matmul(W, input)+b;
        output = std::make_shared<Tensor<T>>(input->rows, this->output_feature);
        output->allocate();
        const int num_per_thread = 1;
        const int block_size_x = 16;
        const int block_size_y = 16;
        int grid_size_x = (output->cols + block_size_x*num_per_thread - 1) / (block_size_x*num_per_thread);
        int grid_size_y = (output->rows + block_size_y*num_per_thread - 1) / (block_size_y*num_per_thread);
        int M = input->rows;
        int N = W->rows;
        int K = input->cols;
        if(input->cols!=W->cols){
            std::ostringstream oss;
            oss << "LinearLayer error: x1->cols != x2->rows which " 
                << input->cols << " != " << W->cols;
            throw std::runtime_error(oss.str()); // 抛出异常并传递格式化后的错误消息
        }
        linearKernel<T, 16, 1><<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(input->data_device.get(), W->data_device.get(), b->data_device.get(), output->data_device.get(), M, N, K);
        NNException::throwIfDeviceErrorsOccurred("linearKernel error\n");
        if(this->train){
            output->op = "linear";
            output->prev.insert(input);
            output->prev.insert(W);
            output->prev.insert(b);
            output->setBackward([this](std::shared_ptr<Tensor<T>> output) {
                this->input->addGrad(mul_(output->grad.get(), this->W.get()));
                this->W->addGrad(Tmul_(output->grad.get(), this->input.get()));
                this->b->addGrad(sum_(output->grad.get(), 1));
            });
        }
        return output;

    }
    void initWeights(unsigned long long seed=0){
        // W.randomInitDevice(seed);
        W->init_Xavier(seed);
        // W->randomInitDevice(seed);
        // W*=0.01;
    }
    void initBias(){
        // b->init_Xavier(89212);
        b->zeroInitDevice();
    }
    void init(){
        float bound = 1/sqrt(this->input_feature);
        uniform_(W->data_host.get(), this->output_feature*this->input_feature, -bound, bound, &rng);
        // for(int i = 0; i<this->input_feature; i++){
        //     std::cout<<W->data_host.get()[i] << " ";
        // }
        // std::cout<<std::endl;
        uniform_(b->data_host.get(), this->output_feature, -bound, bound, &rng);
        // for(int i = 0; i<this->output_feature; i++){
        //     std::cout<<b->data_host.get()[i] << " ";
        // }
        // std::cout<<std::endl;
        W->copyHostToDevice();
        b->copyHostToDevice();
        //初始化w
        // for (int i = 0; i < this->output_feature; i++) {
        //     for(int j = 0; j< this->input_feature; j++){
        //         W->data_host.get()[i*this->input_feature+j] = 
        //     }
    }
    void save_data() {
        W->save_to_file("W.bin");
        b->save_to_file("b.bin");
    }

};
