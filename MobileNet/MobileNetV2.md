### MobileNetV2

##### 主要改进点:
* 引入残差结构，先升维再降维，增强梯度的传播，显著减少推理期间所需的内存占用`(Inverted Residuals)`
* 去掉 Narrow layer（low dimension or depth） 后的 ReLU，保留特征多样性，增强网络的表达能力`(Linear Bottlenecks)`
* 网络为全卷积的，使得模型可以适应不同尺寸的图像；使用 RELU6（最高输出为 6）激活函数，使得模型在低精度计算下具有更强的鲁棒性
* MobileNetV2 building block 如下所示，若需要下采样，可在 DWise 时采用`步长为 2` 的卷积；小网络使用小的扩张系数（expansion factor），
大网络使用大一点的扩张系数（expansion factor），推荐是5~10，论文中 t=6。
![](https://img-blog.csdn.net/20181011141302981)

##### 和MobileNetV1的区别:
![](https://img-blog.csdn.net/20181011145544730)

##### 和ResNet的区别：
![](https://img-blog.csdn.net/2018101114564733)
