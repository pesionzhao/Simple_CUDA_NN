#pragma once
#include"../layers/softmax.cuh"
#include"loss.cuh"
//一维向量的交叉熵
template<typename T>
__global__ void CEforward(T* pre, T* tar, T* res, int N){
    T sum = 0;
    int cols_per_thread = (N+31)/32;
    for(int i = 0; i<cols_per_thread; i++){
        int col = 32*i+threadIdx.x;
        if(col<N){
            T y_i = tar[col];
            T y_hat = pre[col];
            y_hat = y_hat ? y_hat:1e-1;
            sum += -y_i*log(y_hat);
        }
    }
    sum = warpReduce<SumOp, T, 32>(sum);
    if(threadIdx.x == 0) {
        res[0] = sum;
    }
}
template <typename T>
class CE : public Loss<T>{
public:
    float cost(std::shared_ptr<Matrix<T>> predictions, std::shared_ptr<Matrix<T>> target){
        //TODO 形状判断 if(pre->shape!=tar->shape) throw

        //假设输入未作softmax
        std::shared_ptr<Softmax<T>> softmax = std::make_shared<Softmax<T>>();
        this->predictions = softmax->forward(predictions);
        this->target = target;
        //两个一维向量做交叉熵损失
        int block_size = 32;
        int grid_size = 1;
        T* res_device = nullptr;
        cudaMalloc((void**)&res_device, sizeof(T));
        CEforward<T><<<grid_size, block_size>>>(predictions->data_device.get(), target->data_device.get(), res_device, predictions->rows);
        T* res_host = new T;
        cudaMemcpy(res_host, res_device, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(res_device);
        std::cout<<"CE  loss is " <<*res_host<< std::endl;
        return *res_host;
    }
    //针对带有softmax的CE, dL = y_hat - y
    std::shared_ptr<Matrix<T>> dCost(){
        std::shared_ptr<Matrix<T>> dL = std::make_shared<Matrix<T>>(this->predictions->rows, this->predictions->cols);
        *dL = *this->predictions - *this->target;
        return dL;
    }
};