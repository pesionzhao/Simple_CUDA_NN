CUDANN

本项目旨在实现cuda完成基本的网络训练流程，以达到熟悉cuda编程的目标

首先实现矩阵乘法

目前已经完成

**tensor类的实现**

- 初始化： 正态分布，Xavier, 全零初始化
- 运算符重载： 矩阵之间的逐元素计算或者矩阵与标量之间的逐元素计算，以及右移运算符重载
- 梯度的保存

**layer的实现**

线性层: $Y = WX + b$

**优化器的实现**

- 梯度下降
- Adam

**loss的实现**

- MSELoss


TODO

- Tensor的内部变量rows, cols打包为shape
- 可以控制Tensor在cpu上计算或者在gpu上计算
- 线性层的矫正 $Y = XW^T + b$
- 运算符重载的完善, 并考虑基础运算符的backward


TODO

- 只要张量计算，就要存雅可比矩阵
- 怎么构建计算图