#pragma once
#include "nn_layer.h"

template<typename T>
__global__ void reluForward(T* src, T* dst, int M, int N){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M&&col<N){
        dst[row*N+col] = max(0.0, src[row*N+col]);
    }   
}

template<typename T>
__global__ void reluBackwardKernel(T* relu_output, T* dy, T* dx, int M, int N){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int idx = row*N+col;
    if(row<M&&col<N){
        dx[idx] = (relu_output[idx]>0)*dy[idx];
    }   
}

// 只有大于零的值才有梯度
template<typename T>
std::shared_ptr<Matrix<T>> reluBackward(std::shared_ptr<Matrix<T>> relu_output, std::shared_ptr<Matrix<T>> dy){
    int M = dy->rows;
    int N = dy->cols;
    const int block_x = 16;
    const int block_y = 16;
    int grid_x = (dy->cols+block_x-1)/block_x;
    int grid_y = (dy->rows+block_y-1)/block_y;
    std::shared_ptr<Matrix<T>> dx = std::make_shared<Matrix<T>>(dy->rows, dy->cols, false);
    reluBackwardKernel<<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(relu_output->data_device.get(), dy->data_device.get(), dx->data_device.get(), M, N);
    return dx;
}
template<typename T>
class Relu: public NNLayer<T>{
    // std::shared_ptr<Matrix<T>> input;
    std::shared_ptr<Matrix<T>> output;
    //y = max(0, x);
    std::shared_ptr<Matrix<T>> forward(std::shared_ptr<Matrix<T>> input){
        int M = input->rows;
        int N = input->cols;
        std::shared_ptr<Matrix<T>> output = std::make_shared<Matrix<T>>(M, N, input->requires_grad);
        dim3 block_size(16,16);
        int grid_size_x = (input->cols + block_size.x - 1) / (block_size.x);
        int grid_size_y = (input->rows + block_size.y - 1) / (block_size.y);
        reluForward<<<dim3(grid_size_x, grid_size_y), block_size>>>(input->data_device.get(), output->data_device.get(), M, N);
        if(this->train){
            output->op = "relu";
            output->prev.insert(input);
            output->setBackward([input](std::shared_ptr<Matrix<T>> dst){
                input->addGrad(reluBackward(dst, dst->grad));
            });
        }
        return output;
    }
	std::shared_ptr<Matrix<T>> backward(std::shared_ptr<Matrix<T>> dy){
        return nullptr;
    }
};