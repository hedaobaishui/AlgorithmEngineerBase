<!-- TOC -->

- [0. 基础知识](#0-基础知识)
  - [0.1 计算量](#01-计算量)
  - [0.2 参数量](#02-参数量)
- [１.卷积](#１卷积)
  - [1.0 dilated convolution\空洞卷积](#10-dilated-convolution空洞卷积)
  - [1.1 分组卷积](#11-分组卷积)
  - [1.2　深度可分离卷积](#12深度可分离卷积)
  - [1.3卷积后的输出](#13卷积后的输出)
  - [1.4 卷积过程](#14-卷积过程)
- [2.激活函数](#2激活函数)
- [3. LOSS 函数](#3-loss-函数)
- [4. 优化函数](#4-优化函数)
- [5. 学习率调整](#5-学习率调整)
- [3.batchNormal](#3batchnormal)
- [4.参数初始化](#4参数初始化)
- [5.nn.Sequential](#5nnsequential)
- [6.nn.Module](#6nnmodule)
- [7. 逆卷积ConvTranspose2d!](#7-逆卷积convtranspose2d)
- [7. 逆卷积ConvTranspose2d!](#7-逆卷积convtranspose2d-1)
- [7. SyncBatchNorm](#7-syncbatchnorm)
- [8. 卷积计算过程](#8-卷积计算过程)
- [8. 计算过程](#8-计算过程)
- [9. 不同的IOU](#9-不同的iou)
- [10.upsample pixelshuffle](#10upsample-pixelshuffle)

<!-- /TOC -->
# 0. 基础知识
## 0.1 计算量
参数量：
CNN:
一个卷积核的参数 = k*k*Cin+1
一个卷积层的参数 = (一个卷积核的参数)*卷积核数目=k*k*Cin*Cout+Cout

FLOPS:
注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。

FLOPs:
注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。
$$ (2*C_{i}*K^2-1)*H*W*C_{0}$$
## 0.2 参数量
一个卷积核的参数：
$$k*k*C_{in}$$
一个卷积层的参数：
$$k*k*C_{in}*C_{out}+C_{out}$$
# １.卷积
## 1.0 dilated convolution\空洞卷积
* 1.用于图像分割
* 2.up-sampling 和 pooling layer 的设计缺陷   
* 3.asd  
## 1.1 分组卷积
* Alex认为group conv的方式能够增加 filter之间的对角相关性,而且能够减少训练参数,不容易过拟合,这类似于正则的效果。
* 降低计算量:是普通卷积计算量1/group_nums
## 1.2　深度可分离卷积
    *深度可分离卷积是MobileNet的精髓,它由deep_wise卷积和point_wise卷积两部分组成。而深度可分离卷积是进行了两次卷积操作,第一次先进行deep_wise卷积(即收集每一层的特征),kernel_size = K*K*1,第一次卷积总的参数量为K*K*Cin,第二次是为了得到Cout维度的输出,kernel_size = 1*1*Cin,第二次卷积总的参数量为1*1*Cin*Cout。第二次卷积输出即为深度可分离卷积的输出。      
    *分组卷积只进行一次卷积(一个nn.Conv2d即可实现),不同group的卷积结果concat即可
    *举个例子比较参数量：假设input.shape = [c1, H, W] output.shape = [c2, H, W] (a)常规卷积参数量=kernel_size * kernel_size * c1 * c2 (b)深度可分离卷积参数量=kernel_size * kernel_size *c1 + 1*1*c1*c2
## 1.3卷积后的输出
   * 卷积
$$o=\frac{i+2p-k}{s}+1$$
   * 反向卷积
        i为输入尺寸，o为输出尺寸
      * 如果(o+2p-k)%s=0
        o = s(i-1)- 2p + k
      * 如果(o+2p-k)%s!=0
        o = s(i-1)- 2p + k + (o+2p-k)%s
## 1.4 卷积过程
　卷积操作是低效操作，主流神经网络框架都是通过im2col+矩阵乘法实现卷积，以空间换效率。输入中每个卷积窗口内的元素被拉直成为单独一列，这样输入就被转换为了H_out * W_out列的矩阵(Columns)，im2col由此得名；将卷积核也拉成一列后(Kernel)，左乘输入矩阵，得到卷积结果(Output)。im2col和矩阵乘法见如下两图(图片来源：附录1)。
![卷积过程](https://pic4.zhimg.com/80/v2-d6246d784e600ee66945b3bb58dc75e7_720w.jpg)

[数学推导](https://www.cnblogs.com/pinard/p/6494810.html)

asd [参考文献](https://www.jianshu.com/p/f743bd9041b3)    

# 2.激活函数
self.relu = nn.ReLU(inplace=True) inplace 是够在原对象基础上进行修改
# 3. LOSS 函数
* the input :math:`x` and target :math:`y`.
* L1 LOSS
CrossEntropyLoss
$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right| $$
example:

        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
* NLLLoss
$$        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\}$$


$$
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}
        $$
* 
$$        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right]$$
$$        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}
        $$
$$       \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
        l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
        + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right]
        $$
# 4. 优化函数

# 5. 学习率调整

# 3.batchNormal
    nn.BatchNorm2d(num_features) num_featurs:输入数据的通道数量，在几个维度上计算
    在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定，BatchNorm2d()函数数学原理如下：
![归一化公式](https://img-blog.csdnimg.cn/20190612205637399.png)
# 4.参数初始化
[参数初始化](https://blog.csdn.net/longrootchen/article/details/105650059)

# 5.nn.Sequential
    一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
# 6.nn.Module 

是所有神经网络单元的基类
* nn.Module 是所有神经网络单元（neural network modules）的基类
* pytorch在nn.Module中，实现了__call__方法，而在__call__方法中调用了forward函数。
# 7. 逆卷积ConvTranspose2d!
# 7. 逆卷积ConvTranspose2d!
# 7. SyncBatchNorm
# 8. 卷积计算过程
# 8. 计算过程
# 9. 不同的IOU
 IoU、GIoU、DIoU 和 CIoU
 [参考](https://zhuanlan.zhihu.com/p/94799295)
# 10.upsample pixelshuffle
torch.nn.visionlayers
        #改变图像属性　亮度　对比度　饱和度　色调
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)