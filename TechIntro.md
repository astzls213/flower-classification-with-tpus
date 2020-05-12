# 技术简介：

以下纯个人见解，难免有错误，还请读者独立思考并提出质疑。

## 模型，层类：

### （1）GloBalAveragePooling（GAP）层：

使用GAP而不是FC能够提升网络性能的原因主要有两个。

一是从feature maps 到 nodes 不需要学习什么参数，只需将每一个feature map的每个值sum并average即可得到对应的node，加快学习速度，减少参数，缓解过拟合。

二是更加适合空间不变性这个性质。如同卷积层，无论特征在图片的哪里，通过滤波器的平移总能找到该特征。GAP也是一样，他是对特征图所有元素求和，那么特征图的核心特征无论是在哪，总能被运算进去，这提高了分类的健壮性。

三是GAP后的node数量比FC要少很多很多，有效防止分类器的过拟合，

但GAP也有缺点，就是会减慢网络的收敛速度，通过训练结果可以看出。

### （2）Dropout层：

通过简单的断开某个node和另一个node的连接，能够有效抑制过拟合。可以想像Dropout充当的角色就是噪声，就好像在训练图片挖掉某些像素点（但不能挖的太多，否则失去关键特征），但人脑依然能够识别出来一样。这其实就是变相的增加了图片集的数量，所以有效的抑制过拟合，使得网络更好的识别关键特征。

### （3）Batch Normalization（BN）层：

BN层解决的问题主要是梯度传播问题，在训练时，保存来自前一层数值的均值和方差，并假设这些数值的分布服从正态分布，对这些数值进行标准化，再将数值输入到激活层（如RELu）。这样是为了增加激活值的广度，这不仅可以解决梯度消失，爆炸的问题（激活值不会偏向某个值，那么导数就不会偏向某个值），还解决了网络“表现力”受限的问题（如果激活值集中为某个值，那么也就是说很多node都输出相同的某个值，那么要这么这样的node干什么？？用一个这样的node不就行了），使得学习速度提升（误差更好的向前层传播）

### （4）卷积与池化

为什么不把图片展开成1维向量？是因为展开后节点个数太多，导致维度太高，学习很慢。

而且，他没有很好的利用空间对特征的影响。而卷积，恰恰就是由此诞生。其实卷积是一个术语，应该是来自图像处理的下采样。通过设置不同的滤波器（矩阵）对图片矩阵进行平移运算，得到特征图，这个过程称之为卷积。

池化就是对特征图的一个降维，通常就是取一个范围内的最大值，然后转成新的矩阵，这个新矩阵比之前的矩阵更小，并且更能反映出特征（抗噪）。

具体原理就不多说了，网上很多说法。

## 优化方法类：

### （1）Adam优化法：

实际上，Adam就是Adagrad和Momentum的缝合版，为什么呢？

回想一下，Momentum优化实际做的事是：设å为学习率，ß为摩擦因子，g<sub>i</sub>为梯度（其中i=epoch轮次）。那么，v<sub>i</sub> = ß<sub>1</sub> * v<sub>i-1</sub> + (1-ß<sub>1</sub>)*g<sub>i-1</sub>然后，W<sub>i</sub> = W<sub>i-1</sub> + v<sub>i</sub> 也就是说权值每一次的更新，都和历史梯度值有关，随着epoch推移，越早epoch得到的梯度值对更新的贡献就越小（摩擦力）

那么Adagrad又做了什么呢？实际上，AdaGrad的核心目的就是减缓“学习率”，就是给当前å再乘上一个系数因子p，这个p = 1 / sqrt(h), 其中h = ∑ (g<sub>i</sub> * g<sub>i</sub>), i为epoch。也就是说，p是历史梯度的平方的 求和 的 开方 的 倒数，这样随着时间推移，h会越来越大，而p就会越来越小，值得注意的是，p是一个矩阵，也就是说，权值梯度大的，p对应的系数就越小，那么对这个梯度大的权值更新就小，那么就可以缓解波动的情况。

那么，什么是Adam呢？其实，AdaGrad时，h只是简单的把所有历史梯度的平方求和，如果，我们像Momentum给他整个摩擦因子ß，使得h<sub>i</sub> = ß*h<sub>i-1</sub> +∂(g<sub>i</sub> * g<sub>i</sub>) ,其中∂是当前梯度平方贡献因子，通常为1-ß。那么h就会记住一定范围内的梯度平方和，而不是整个epochs的梯度平方和，这样可以避免，p收敛到0导致学习停止的情况。

那么总结一下，Adam的更新公式就是：v<sub>i</sub> = ß<sub>1</sub> * v<sub>i-1</sub> + (1-ß<sub>1</sub>)*g<sub>i-1</sub> 然后，W<sub>i</sub> = W<sub>i-1</sub> +å * p * v<sub>i</sub> 

## 超参数调整类：

### （1）网格搜索学习率lr：

通过随机出不同的学习率，然后对这些学习率进行fitting，看看哪个学习率效果比较好，然后基于此缩小范围，再生成一堆学习率，再以此类推。。

## 参考文献：

[1]"Network In Network", by Min Lin, Qiang Chen, Shuicheng Yan, https://arxiv.org/abs/1312.4400

[2]"Improving neural networks by preventing co-adaptation of feature detectors", by Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov,  https://arxiv.org/pdf/1207.0580.pdf

[3]“Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”, by Sergey Ioffe, Christian Szegedy,  https://arxiv.org/pdf/1502.03167.pdf

[4]"Adam: A Method for Stochastic Optimization", by Diederik P. Kingma, Jimmy Ba, https://arxiv.org/pdf/1412.6980.pdf
