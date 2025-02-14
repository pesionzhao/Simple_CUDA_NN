#pragma once
#include <memory> //智能指针
#include <random> //随机初始化
#include <iomanip> //保留位数
#include <iostream>
#include <curand_kernel.h>//随机初始化
#include <unordered_set>
#include "nn_exception.cuh" //异常
#include "../kernel/kernel.cuh"
#include "../kernel/base.cuh"
#include "rand.h"

extern mt19937_state rng;
mt19937_state rng;

enum InitType
{
    ZERO,
    RANDOM
};
//随机初始化data_device
template<typename T>
__global__ void randomInitKernal(T* data, int rows, int cols, unsigned long long seed = 0)
{
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx_y*cols+idx_x;
    curandState state;
    curand_init(seed, index, 0, &state);
    if (idx_x<cols&&idx_y<rows) {
        // 均匀分布
        // data[idx_y*cols+idx_x] = static_cast<T>(curand_uniform_double(&state))/cols;
        // 高斯分布
        data[idx_y*cols+idx_x] = static_cast<T>(curand_normal_double(&state));
    }
}
//全零初始化data_device
template<typename T>
__global__ void zeroInitKernal(T* data, int rows, int cols)
{
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x<cols&&idx_y<rows) {
        data[idx_y*cols+idx_x] = 0;
    }
}

template<typename T>
class Tensor {
private:
    bool device_allocated = false;
    bool host_allocated = false;
    bool in_cuda = true;
    void allocate_device(int N){
        if(!device_allocated){
            T* device_ptr = nullptr; //开辟空间在cudaMalloc实现
            cudaMalloc((void**)&device_ptr, sizeof(T)*rows*cols);
            NNException::throwIfDeviceErrorsOccurred("cudaMalloc failed\n");
            //使用智能指针管理
            data_device = std::shared_ptr<T>(device_ptr, [](T* p) {cudaFree(p);});//定义删除器
            device_allocated = true;
            //TODO cuda or cpu
            if(requires_grad){
                grad = std::make_shared<Tensor<T>>(shape, false);
                grad->allocate_device(N);
                grad->zeroInitDevice();
            }
        }
    };
    void allocate_host(int N){
        if(!host_allocated){
            T* host_memory = new T[N];
            data_host = std::shared_ptr<T>(host_memory, [](T* p) {delete[] p;});
            host_allocated = true;
        }
    };

public:
    bool requires_grad = true;
    std::shared_ptr<T> data_device;
    std::shared_ptr<T> data_host;
    std::shared_ptr<Tensor<T>> grad;
    std::unordered_set<std::shared_ptr<Tensor<T>>> prev; //前驱节点
    std::function<void(std::shared_ptr<Tensor<T>>)> grad_fn = [this](std::shared_ptr<Tensor<T>>) { };; //反向传播函数
    std::string op = "";
    std::vector<int> shape;  // shape.size()==1时为向量， shape.size()>1时为矩阵
    int rows;
    int cols;
    Tensor(){};
    Tensor(int len){
        shape = {len};         
        rows = shape[0];
        cols = 1;
        allocate();
    };
    Tensor(std::vector<int>& shape, bool requires_grad = true) : requires_grad(requires_grad){
        this->shape=shape;
        rows = shape[0];
        cols = shape[1];
        allocate();
    };
    Tensor(std::vector<int>&& shape, bool requires_grad = true) : requires_grad(requires_grad){
        this->shape=std::move(shape);
        rows = shape[0];
        cols = shape[1];
        allocate();
    };
    //默认行向量
    Tensor(int featrues, bool requires_grad=true) : cols(featrues), requires_grad(requires_grad){shape = {1, cols}; allocate();};
    Tensor(int rows, int cols) : rows(rows), cols(cols){shape = {rows, cols}; allocate();};
    Tensor(int rows, int cols, InitType init) : rows(rows), cols(cols){
        shape = {rows, cols}; 
        allocate();
        switch (init){
        case ZERO:
            zeroInitDevice();
            NNException::throwIfDeviceErrorsOccurred("zeroInitDevice error!\n");
            break;
        case RANDOM:
            randomInitDevice(static_cast<unsigned long long>(time(0)));
            NNException::throwIfDeviceErrorsOccurred("randomInitDevice error!\n");
            break;
        default:
            break;
        }
    };
    Tensor(int rows, int cols, bool requires_grad) : rows(rows), cols(cols), requires_grad(requires_grad){shape = {rows, cols};allocate();};
    //深拷贝
    Tensor<T> clone(){
        Tensor<T> copy(rows, cols, requires_grad);
        allocate();
        int block_size = 64;
        int grid_size = (rows*cols+block_size-1)/block_size;
        copyKernel<<<grid_size, block_size>>>(data_device.get(), copy.data_device.get(), rows*cols);
        NNException::throwIfDeviceErrorsOccurred("Copy error!\n");
        return copy;
    };
    void allocate(){
        int N = 1;
        for(int i = 0; i<shape.size(); ++i){
            N*=shape[i];
        }
        allocate_device(N);
        allocate_host(N);
    };
    void copyHostToDevice(){
        //此时要保证指针已被初始化
        if(host_allocated&&device_allocated){
            cudaMemcpy(data_device.get(), data_host.get(), sizeof(T)*rows*cols, cudaMemcpyHostToDevice);
            NNException::throwIfDeviceErrorsOccurred("cudaMemcpy HostToDevice failed\n");
        }
        else{
            throw NNException("Cannot copy host data to not allocated memory on device.\n");
        }
    };
    void copyDeviceToHost(){
        //此时要保证指针已被初始化
        if(host_allocated&&device_allocated){
            cudaMemcpy(data_host.get(), data_device.get(), sizeof(T)*rows*cols, cudaMemcpyDeviceToHost);
            NNException::throwIfDeviceErrorsOccurred("cudaMemcpy DeviceToHost failed\n");
        }
        else{
            throw NNException("Cannot copy device data to not allocated memory on host.\n");
        }
    };
    //随机初始化data_host
    void randomInitHost(){
        float bound = 1;
        uniform_(data_host.get(), this->rows*this->cols, -bound, bound, &rng);
        // 初始化随机数生成器
        // std::random_device rd;
        // std::mt19937 gen(42);  // 使用随机设备作为种子
        // std::cout<<"cols = " <<cols<< std::endl;
        // std::cout<<"rows = " <<rows<< std::endl;
        // T bound = std::sqrt(6.0 / cols);
        // std::uniform_real_distribution<> dis(-bound, bound);
        // // std::normal_distribution<> dis(-bound, bound);

        // // 填充矩阵
        // for (int i = 0; i < rows; ++i) {
        //     for (int j = 0; j < cols; ++j) {
        //         data_host.get()[i*cols+j] = dis(gen);  // 使用分布生成随机数
        //     }
        // }
    };
    //全零初始化data_host
    void zeroInitHost(){
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data_host.get()[i*cols+j] = 0;
            }
        }
    };
    //生成N(0，1)的高斯分布
    void randomInitDevice(unsigned long long seed = 0){
        int block_size_x = 16;
        int block_size_y = 16;
        int grid_size_x = (cols + block_size_x - 1) / block_size_x;
        int grid_size_y = (rows + block_size_y - 1) / block_size_y;
        randomInitKernal<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(data_device.get(), rows, cols, seed);
        NNException::throwIfDeviceErrorsOccurred("randomInitKernal failed\n");
        in_cuda = true;
    };
    // Xavier初始化
    void init_Xavier(unsigned long long seed = 0){
        int block_size_x = 16;
        int block_size_y = 16;
        int grid_size_x = (cols + block_size_x - 1) / block_size_x;
        int grid_size_y = (rows + block_size_y - 1) / block_size_y;
        randomInitKernal<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(data_device.get(), rows, cols, seed);
        *this *= (T)(1/sqrt(rows));
    }
    //均匀分布初始化
    void init_Uniform(unsigned long long seed = 0){

    }
    // 全零初始化
    void zeroInitDevice(){
        int block_size_x = 16;
        int block_size_y = 16;
        int grid_size_x = (cols + block_size_x - 1) / block_size_x;
        int grid_size_y = (rows + block_size_y - 1) / block_size_y;
        zeroInitKernal<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(data_device.get(), rows, cols);
        NNException::throwIfDeviceErrorsOccurred("randomInitKernal failed\n");
        in_cuda = true;
    };
    //定义反向传播函数指针
    void setBackward(const std::function<void(std::shared_ptr<Tensor<T>>)>& func) {
        grad_fn = func;
    }
    T& operator[](int index) {
        return data_host.get()[index];
    };
    //梯度累加
    void addGrad(std::shared_ptr<Tensor<T>> other_grad){
        grad = grad+other_grad;
    }
    //运算符重载，不计算梯度, 计算梯度的函数在launch.h，智能指针的运算符重载 
    //Tensor+Tensor
    template<typename U>
    Tensor<T> operator+(const Tensor<U>& other){
        if(this->shape!=other.shape)
            throw NNException("+ operator catch the different shape");
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        add<<<grid_size, block_size>>>(data_device.get(), other.data_device.get(), dst->data_device.get(), rows, cols);
        return *dst;
    }
    //Tensor+scalar
    Tensor<T> operator+(const T scalar){
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        addScalar<<<grid_size, block_size>>>(data_device.get(), scalar, dst->data_device.get(), rows, cols);
        return *dst;
    }
    //一元负号运算符: -Tensor
    Tensor<T> operator-(){
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        opposite<<<grid_size, block_size>>>(data_device.get(), dst->data_device.get(), rows, cols);
        return *dst;
    }
    //Tensor-Tensor
    template<typename U>
    Tensor<T> operator-(const Tensor<U>& other){
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        minus<<<grid_size, block_size>>>(data_device.get(), other.data_device.get(), dst->data_device.get(), rows, cols);
        return *dst;
    }
    //Tensor*scalar (broadcast)
    template<typename U>
    Tensor<T> operator*(U scalar) {
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        mulScalar<<<grid_size, block_size>>>(data_device.get(), scalar, dst->data_device.get(), rows, cols);
        return *dst;
    };
    //Tensor*Tensor (element-wise)
    template<typename U>
    Tensor<T> operator*(const Tensor<U>& other) {
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        mulElement<<<grid_size, block_size>>>(data_device.get(), other.data_device.get(), dst->data_device.get(), rows, cols);
        return *dst;
    }
    //Tensor/scalar (broadcast)
    template<typename U>
    Tensor<T> operator/(const U scalar){
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        divScalar<<<grid_size, block_size>>>(data_device.get(), scalar, dst->data_device.get(), rows, cols);
        return *dst;
    }
    //Tensor/scalar (element-wise)
    template<typename U>
    Tensor<T> operator/(const Tensor<U>& others){
        std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(rows, cols, requires_grad);
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        divElement<<<grid_size, block_size>>>(data_device.get(), others.data_device.get(), dst->data_device.get(), rows, cols);
        return *dst;
    }
    //重载*=运算符，用作标量乘矩阵
    template<typename U>
    Tensor<T>& operator*=(U scalar) {
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        mulScalarAssgin<<<grid_size, block_size>>>(data_device.get(), scalar, rows, cols);
        return *this;
    };
    // Tensor<T> T_() {
    //     std::shared_ptr<Tensor<T>> dst = std::make_shared<Tensor<T>>(cols, rows, requires_grad);
    //     dim3 block_size(16,16);
    //     dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
    //     transpose<<<grid_size, block_size>>>(data_device.get(), dst->data_device.get(), rows, cols);
    //     return *dst;
    // };
    Tensor<T> sqrt_() {
        dim3 block_size(16,16);
        dim3 grid_size((cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1)/ block_size.y);
        sqrtElement<<<grid_size, block_size>>>(data_device.get(), rows, cols);
        return *this;
    };
    //用于调试的函数， functions only for debug
    //重载右移运算符用于输出 cout << Tensor
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& Tensor){
        size_t x = Tensor.rows;
        size_t y = Tensor.cols;
        os << "Tensor (" << x << ", " << y << "):" << std::endl;
        os << std::fixed <<std::scientific<< std::setprecision(2);
        // 打印矩阵内容，假设host已经有数据
        for (size_t i = 0; i < x; ++i) {

            for (size_t j = 0; j < y; ++j) {
                os <<  std::setw(5) << Tensor.data_host.get()[i * y + j] << " ";
            }
            os << std::endl;
        }
        os<<std::defaultfloat;
        return os;
    };
    //将矩阵数据保存到文件，用于和pytorch比较
    void save_to_file(const std::string& filename) {
        if(in_cuda)
            copyDeviceToHost();
        FILE* file = fopen(filename.c_str(), "wb");
        if (file == nullptr) {
            std::cerr << "Error opening file " << filename << std::endl;
            return;
        }
        fwrite(data_host.get(), sizeof(T), rows * cols, file);
        fclose(file);
    };
    //判断两个矩阵是否相等
    void compare(Tensor<T>& other) {
        int flag = 0;
        for (int i = 0; i < rows * cols; i++) {
            if (data_host.get()[i] != other.data_host.get()[i]) {
                std::cout << "Mismatch at index " << i << ": " << data_host.get()[i] << " vs " << other.data_host.get()[i] << std::endl;
                flag = 1;
            }
        }
        if (flag == 0) {
            std::cout << "Matrices match" << std::endl;
        }
    };
};