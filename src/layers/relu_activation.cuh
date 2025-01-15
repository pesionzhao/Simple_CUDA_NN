#include "nn_layer.h"

template<typename T>
__global__ void reluForward(T* src, T* dst, int M, int N){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M&&col<N){
        dst[row*N+col] = max(0, src[row*N+col]);
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

//只有大于零的值才有梯度
template<typename T>
std::shared_ptr<Matrix<T>> reluBackward(std::shared_ptr<Matrix<T>> relu_output, std::shared_ptr<Matrix<T>> dy){
    int M = dy->rows;
    int N = dy->cols;
    const int block_x = 16;
    const int block_y = 16;
    int grid_x = (dy->cols+block_x-1)/block_x;
    int grid_y = (dy->rows+block_y-1)/block_y;
    std::shared_ptr<Matrix<T>> dx = std::make_shared<Matrix<T>>(dy->rows, dy->cols, false);
    reluBackward<<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(relu_output->data_device.get(), dy->data_device.get(), dx->data_device.get(), M, N);
    return dx;
}

template<typename T, int block_size=32, int num_per_thread=4>
__global__ void mulKernel(const T* W, const T* A, T* Y, int M, int N, int K) {
    int indA = num_per_thread*blockIdx.y * blockDim.y;
    int indB = num_per_thread*blockIdx.x * blockDim.x;
    //一个线程搬移一块内存，所以要进行线程重排，说白了就是把TILE变小，从而可以处理更多的元素
    int TILE = block_size/num_per_thread;
    __shared__ T SW[block_size*block_size];
    __shared__ T SA[block_size*block_size];
    int idx = threadIdx.x +threadIdx.y * blockDim.x;
    int num_packs = (K + TILE - 1)/TILE;
    int SW_x = idx%TILE;
    int SW_y = idx/TILE;
    int SA_x = idx%(block_size*num_per_thread);
    int SA_y = idx/(block_size*num_per_thread);
    T sum[num_per_thread*num_per_thread] = {0.0f};
    for (int i = 0; i < num_packs; ++i) {
        SW[idx] = indA+SW_y<M&&TILE*i+SW_x<K? W[(indA+SW_y)*K+TILE*i+SW_x]:0;
        SA[idx] = i*TILE+SA_y<K&&SA_x+indB<N? A[(i*TILE+SA_y)*N+SA_x+indB]>0;
        __syncthreads();
        for (int j = 0; j < num_per_thread*num_per_thread; ++j){
            for(int t = 0; t < TILE; ++t){
                sum[j] += SW[t+(j/num_per_thread+threadIdx.y*num_per_thread)*TILE]*SA[t*block_size*num_per_thread+threadIdx.x*num_per_thread+j%num_per_thread];
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < num_per_thread*num_per_thread; ++j){
        int row = indA + threadIdx.y*num_per_thread+j/num_per_thread;
        int col = indB + threadIdx.x*num_per_thread+j%num_per_thread;
        if(row<M&&col<N)
        {
            Y[row*N+col] = sum[j];
        }
    }
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
            output->setBackward([this](std::shared_ptr<Matrix<T>> dy){
                output->addGrad(reluBackward(this->output, dy);)
            })
        }
    }
	std::shared_ptr<Matrix<T>> backward(std::shared_ptr<Matrix<T>> dy)
    [
        return nullptr;
    ]
};