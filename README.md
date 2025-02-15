# CUDA_NN

本项目旨在实现cuda完成基本的网络训练流程，以达到熟悉cuda/c++编程的目标，

目前完成了

- tensor的实现与基础运算符的自动微分，完成了线性层、relu，softmax层的正向模拟与反向传播，损失函数：mse与crossentropy的计算与反传
- 借鉴K神完成了与pytorch的相同的随机数生成器，可以固定一样的权重，用于对比此项目和pytorch的计算
- adam优化器和梯度下降优化器对梯度进行更新
- 可以对mnist数据集进行训练与推理


参考仓库：

https://github.com/leeroee/NN-by-Numpy

https://github.com/Phoenix8215/BuildCudaNeuralNetworkFromScratch

https://github.com/SmartFlowAI/LLM101n-CN/blob/master/micrograd/micrograd.cpp

## **tensor类的实现**
仿照pytorch, 首先要定义tensor, 也是网络运算的基本单元

- 初始化： 正态分布，Xavier, 全零初始化
- 运算符重载以及基本的运算： 矩阵之间的逐元素计算或者矩阵与标量之间的逐元素计算，以及右移运算符重载
- 梯度的保存

目前只实现二维张量, 定义如下（省略版）

```c++
template<typename T>
class Tensor{
public:
    //数据形状
    int rows, cols;
    //数据指针
    std::shared_ptr<T> data_host;
    std::shared_ptr<T> data_device;
    Tensor(int rows, int cols):rows(rows), cols(cols){}{};
    //开辟空间
    void allocate(){};
    //数据转移
    void copyHostToDevice(){};
    void copyDeviceToHost(){};
    //运算符重载
    Tensor operator+(Tensor& other){}
    Tensor operator-(Tensor& other){}
}
```

### layer的forward与backward公式推导

#### 线性层

forward:

$y = Wx + b$ 

backward:

$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial W} = \frac{\partial L}{\partial y} X^T$

$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial b} = \frac{\partial L}{\partial y}$

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = W^T \frac{\partial L}{\partial y}$

这里的左乘右乘可能会一头雾水，说白了就是要对应元素相乘，矩阵的元素用偏导数的形式写出来就会懂了。

pytorch的线性层为什么不是`y=W*x+b` 而是 `y = x*W^T+b`?

说白了就是把列向量当成行向量

行优先符合主流的编程规范，比如sum, max, 所以pytorch将优先的行向量作为特征，行数为批次, 并且按道理存数据时，单个数据内存要连续，如果行数为特征，列数为批次的话，单个数据的不同特征内存不连续

所以本项目实现的是 `y = x*W^T+b` ， 当然作为对比 `y=W*x+b` 也实现了

#### softmax
[反向传播之一：softmax函数](https://zhuanlan.zhihu.com/p/37740860)
[SoftMax反向传播推导，简单易懂，包教包会](https://www.bilibili.com/video/BV143411h7PQ/?spm_id_from=333.337.search-card.all.click&vd_source=c43347ef375755d298da8f0c05cfe444)

forward:

$$y_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

backward:

$$
\frac{\partial y_i}{\partial x_j} = 
\begin{cases} 
y_i - y_i^2, & \text{当 } i = j \\  
-y_i \cdot y_j, & \text{当 } i \neq j \end{cases}
$$

### loss

#### MSE

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

其中：
- $y_i$ 是真实值
- $\hat{y}_i$ 是预测值
- $n$ 是数据点的数量

对 $\hat{y}_i$ 进行求导：

$$
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n} (\hat{y}_i - y_i)
$$

#### CrossEntropy

多类分类问题，对于一个单一样本的交叉熵损失函数，假设真实标签是 $y$（是一个 one-hot 编码的向量），模型输出的概率分布是 $\hat{y}$，交叉熵定义为：

$$L = -\sum_{i=1}^{n}y_i\ln(\hat y_i)$$

对于多类交叉熵损失函数，对预测概率 $\hat{y}_i$ 求导数：

$$
\frac{L(y, \hat{y})}{\partial \hat{y}_i} = - \frac{y_i}{\hat{y}_i}
$$

### optimizer

[十分钟搞明白Adam和AdamW，SGD，Momentum，RMSProp，Adam，AdamW](https://www.bilibili.com/video/BV1NZ421s75D/?spm_id_from=333.337.search-card.all.click&vd_source=c43347ef375755d298da8f0c05cfe444)


遇到的一些坑以及解决方法

1. MSE的反向传播要除以元素个数，否则回传的梯度过大，导致梯度爆炸
2. 线性层的定义注意是Y=XW+b, 为了符合编程行优先，对注意各个参数的偏导数计算
3. 神经网络权重的初始化，正态分布，如果方差过大同样导致梯度爆炸，参考torch的初始化，0均值0.01方差
4. shared_ptr和函数声明周期一样，函数结束自动释放内存，如果要保存就要在其他地方指向这块内存！
5. 在adam更新梯度时，同样进行了tensor的运算，但不保存梯度！！计算更新量时中间变量数量级比较小，所以使用double, 权重参数可能会使用float, 故要进行重载运算符
6. 对于mnist数据集, 如果bs=1，也就是一维一维的进行训练，无法收敛

TODO 

- [ ] 使用CmakeLists.txt对项目进行配置
- [ ] cuda端和cpu端的区分，控制Tensor在cpu上计算或者在gpu上计算
- [ ] 模型的导出与读取
- [ ] softmax二维的backward
- [ ] 更多层的实现
