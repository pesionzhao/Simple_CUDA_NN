//乘法相关
template<typename T, typename U>
__global__ void mulScalar(const T* src, const U scalar, T* dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        T val = src[row * N + col];
        dst[row * N + col] = val * scalar;
    }
}

template<typename T, typename U>
__global__ void mulScalarAssgin(T* src, U scalar, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T val = src[row * N + col];
        src[row * N + col] = val*scalar;
    }
}

template<typename T, typename U>
__global__ void mulElement(const T* src1, const U* src2, T* dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T val1 = src1[row * N + col];
        T val2 = src2[row * N + col];
        dst[row * N + col] = val1 * val2;
    }
}

//除法相关
template<typename T, typename U>
__global__ void divElement(const T* src1, const U* src2, T* dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T val1 = src1[row * N + col];
        T val2 = src2[row * N + col];
        dst[row * N + col] = val1 / val2;
    }
}
template<typename T, typename U>
__global__ void divScalar(const T* src1, const U scalar, T* dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T val1 = src1[row * N + col];
        dst[row * N + col] = val1 / scalar;
    }
}

//加法相关
template<typename T>
__global__ void addScalar(const T* src, const T scalar, T* dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        T val = src[row * N + col];
        dst[row* N + col] = val + scalar;
    }
}

template<typename T>
__global__ void addScalarAssgin(T* src, T scalar, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        T val = src[row * N + col];
        src[row * N + col] = val + scalar;
    }
}

template<typename T, typename U>
__global__ void add(const T* src1, const U* src2, T* dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        T val1 = src1[row * N + col];
        T val2 = src2[row * N + col];
        dst[row* N + col] = val1 + val2;
    }
}

template<typename T, typename U>
__global__ void addAssign(T* src1, U* src2, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        T val1 = src1[row * N + col];
        T val2 = src2[row * N + col];
        src1[row* N + col] = val1 + val2;
    }
}

//减法相关
template<typename T, typename U>
__global__ void minus(T* src1, U* src2, T* dst, int M, int N){
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    if (row < M && col < N){
        T val1 = src1[row * N + col];
        T val2 = src2[row * N + col];
        dst[row* N + col] = val1 - val2;
    }
}

template<typename T>
__global__ void opposite(T* src, T* dst, int M, int N){
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    if (row < M && col < N){
        T val = src[row * N + col];
        dst[row* N + col] = -val;
    }
}

//逐元素根号
template<typename T>
__global__ void sqrtElement(T* src1, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        T val1 = src1[row * N + col];
        src1[row* N + col] = sqrt(val1);
    }
}

//转置
template<typename T>
__global__ void transpose(T* src, T* dst, int M, int N){
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    if (row < M && col < N){
        T val = src[row * N + col];
        dst[row + col*M] = val;
    }
}