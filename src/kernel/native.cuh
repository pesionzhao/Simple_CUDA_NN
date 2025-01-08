template<typename T>
__global__ void mulTKernel_native(const T* W, const T* A, T* Y, int M, int N, int K) {
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    if(row<M&&col<N)
    {
        T sum = 0;
        for(int i=0;i<K;i++)
        {
            sum += W[row*K+i]*A[i+col*K];
        }
        Y[row*N+col] = sum;
    }
};

template<typename T>
__global__ void TmulKernel_native(const T* W, const T* A, T* Y, int M, int N, int K) {
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    if(row<M&&col<N)
    {
        T sum = 0;
        for(int i=0;i<K;i++)
        {
            sum += W[row+i*M]*A[i*N+col];
        }
        Y[row*N+col] = sum;
    }
}

template<typename T>
__global__ void softmax_native(const T* src, T* dst, int M, int N) {
    

}