# My Notes for CS231n
大型混乱系统优化调参现场（需炼丹师对数值优化、物理的极深洞见，对数据的极深insight, imagination, and patience)

## Part1: Basic
### 数据预处理
Normalize the image in each channel of the dataset

### Weight Initialization
何凯明的weight initialization 2/sqrt(Din), 防止网络层加深后, 由于激活函数连续作用产生的梯度弥散

### 模型
1. SVM: linear classifier, 找到几何上对各数据点间隔最大的分割超平面
2. Softmax: linear classifier, normalize score to probability
   $p = exp(e_{s_yi})/sum_j(exp(e_{j}))$, loss $-log(p)$

### BN的作用, 原理, 方法，训练，和预测
1. BatchNormalization的引入是为了解决输入信息落在激活函数激活区间外，导致梯度为0，神经网络参数无法更新，无法再继续训练的问题；引用原paper的话，“if you want a normalized data, just make them so”。
2. Batch Normalization的具体做法分训练时和预测时，训练时，我们针对每一通道(channel)对这个batch的所有samples算平均值和标准差，然后归一化到(-1, 1)的区间内, 并且训练两个超参\gamma和\beta, 分别来拟合将数据还原回去的方程和平均值。预测时我们就使用 \gamma * x + \beta将数据还原回去。
3. 此外还有instance normalization, layer normalization 等方案，我在训练时一般使用Batch Normalization

### 激活函数
1. 作用: increase non-linearity, 让神经网络能够做出更多样的决定
2.	ReLU, LeakyRelu, ELU, Sigmoid,tanh
3.	我训练时用ReLU, 因为简单、快、在正区间梯度不会消失
4.	Sigmoid在饱和区间梯度会消失，且computation-expensive

###	优化器
1. SGD 
随机梯度下降，由于很难对整个训练集算全局的更新梯度, 有这样全局的方法，但很expensive, 我们就采取每次迭代时选取一个mini-batch, 在这个mini-batch算梯度，然后再用梯度乘上learning rate, 来用这个lr*gradient 来更新参数

```python
for iter in range(max_iters):
    x = random_sample(iter)
    dw = caculate_gradient(x)
    w -= lr * dw
```
2. SGD+Momentum
SGD存在的问题: 当我们到达一个local minimum或着saddle point的时候，梯度几乎为0，神经网络就stuck住了, 无法再更新训练，通过加入momentum(借鉴物理模拟中的动量)，我们就可以将之前梯度的信息也考虑进来，再遇到local minimum或者saddle point, 可以借助momentum来跨过这个saddle point

```python
momentum_w = 0
for iter in range(max_iters):
    x = random_sample(iter)
    dw = caculate_gradient(x)
    momentum_w = rho * momentum_w + dw 
    w -= lr * momentum_w 
```
3. Autograd
由于数据集中有很多noise, 用SGD/SGD+Momentum来训练网络, 更新路径会比较曲折, Autograd根据梯度矩阵中每一个元素的模长来自适应地调整更新步长, 使得更新速度在各维度上保持一致, 可以更加平稳得寻找到稳态(Energy minimum)
```python
for iter in range(max_iters):
    x = random_sample(iter)
    dw = caculate_gradient(x)
    w -= lr * dw / (sqrt(dw * dw) + 1e-7)
```

4. RMSProp
根据历史信息调整当前步的更新步长
```python
momentum_w = 0
for iter in range(max_iters):
    x = random_sample(iter)
    dw = caculate_gradient(x)
    momentum_w = beta * momentum_w + (1-beta) * dw * dw
    w -= lr * dw / (sqrt(momentum_w) + 1e-7)
```

5. Adam(My favorite default): alpha = 0.9, beta = 0.99
```python
first_momentum, second_momentum = 0, 0
for iter in range(max_iters):
    x = random_sample(iter)
    dw = caculate_gradient(x)
    first_momentum = alpha * first_momentum + (1-alpha) * dw
    second_momentum = beta * second_momentum + (1-beta) * dw * dw
    w -= lr * first_momentum / (sqrt(second_momentum) + 1e-7)
```
结合历史信息（alpha * first_momentum）和 当前信息 （ (1-alpha) * dw）, 对梯度normalization (first_momentum / (sqrt(second_momentum)), 使得更新步长在整个landscape的各个维度上保持相同的步调, 然后更加稳定地到达稳态
```python
first_momentum, second_momentum = 0, 0
for iter in range(max_iters):
    x = random_sample(iter)
    dw = caculate_gradient(x)
    first_momentum = alpha * first_momentum + (1-alpha) * dw
    second_momentum = beta * second_momentum + (1-beta) * dw * dw
    # to prevent 'divided by zero'/infinity in the first few trainning iterations
    first_unbais = first_moment / (1 - alpha  **t)
    second_unbais = second_moment / (1 - beta  **t)
    w -= lr * first_momentum / (sqrt(second_momentum) + 1e-7)
```

### 如何防止模型过拟合？增加模型泛化能力？
1. regularization 通过在原有loss加上对权重w的`l2_norm loss`,将权重限制在一个很小的区间内
2. dropout 随机抑制一些神经元不让进行梯度传导, dropconnect 随机断掉神经元间的连接不让梯度传导
3. data augmentation (数据增强): affine transformation, crop, color jitterinig, ...

### 如何调整学习速率lr？
1. Step decaying, 在训练80, 120, 150等epochs后降低学习速率, 因为当我们发现学习速度下降后代表着NN已经到达一个minimum, 我们再把lr调低, 以更细的粒度去寻找更优的稳态
2. default initial lr: 1e-3, 5e-4

### 如何训练？
#### Small sample
1. Check initial loss 检测初始的loss是否符合预期
2. Overfitting a small sample 在小训练集上过拟合
#### Large dataset: official trainning
3. Find a lr that makes the loss go down 找到让loss降低的学习速率
4. Coarse grid, train for 1-5 epochs
5. Refine grid, train longer
6. Lookat the loss and accuracy curve
7. goto step 5

### Architecture?
#### One fact: deeper network layers, smaller conv kernel add more non-linearity, but deeper models are harder to optimize

1. Receptive field: two 3x3 conv kernel == 5x5 conv kernel
2. Number parameters: two 3x3 conv kernel < 5x5 conv kernel

#### VGG
#### GoogleNet & MobileNet (减少模型参数)
传统情况下,我们会用 3 x 3 x input_channel x output_channel 的卷积核来卷积

1. depthwise convolution: 每个卷积核只负责卷积一个通道,卷积核与通道一 一 对应, 共有 input_channel 个卷积核
2. pointwise convolution: 用output_channel 个 1 x 1 x input_channel的卷积核在depthwise的输出上跨通道(Cross-channel on-depth)卷积
#### ResNet
1. Residual Block: 残差网络，加入跨层连接，将底层特征inject到更深的layer。use network layers to fit a resiual mapping instead of the direct underlying mapping. Traditional conv: x-->H(x), ResNet: H(x)= F(x) + x, F(x)由神经网络来学习。Use bottleneck layer to improve efficiency.
2. ResNext: 为了让梯度更好的传递, 每一层的residual block由一个变成三个, 
#### FPN 特征金字塔
1. Top-down connection, lateral connection, add more features to learn

## Part2 GAN

## Part3 Object Detection | Instance Segmentation
#### To select ROI, we need NMS suppression
### Two-stage: RCNN, FastRCNN | MaskRCNN
1. Region Proposal: Consume lots of time
2. Select region with highest score
3. Predict Box Offset or Mask on ROI
### One-Stage: YOLO | SOLO
1. Divide the original image into S x S grid
2. Predict a score, box, mask for each grid

## Part4 RNN(Recurent Neural Network) and LSTM
