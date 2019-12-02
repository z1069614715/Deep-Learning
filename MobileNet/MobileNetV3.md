### 高效的模型构建模块

__MobileNetV3 是神经架构搜索得到的模型，其内部使用的模块继承自：__
* MobileNetV1模型引入的深度可分离卷积
* MobileNetV2模型引入的具有线性瓶颈的倒残差结构
* MnasNet模型引入的基于squeeze and excitation结构的轻量级注意力模型

### 互补搜索技术组合

__资源受限的NAS:__
计算和参数量受限的前提下搜索网络的各个模块，所以称之为模块级的搜索
<br><br>
__NetAdapt:__
用于对各个模块确定之后网络层的微调

### 网络结构的改进
![](https://img-blog.csdnimg.cn/20190513145953851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RMX3dseQ==,size_16,color_FFFFFF,t_70)
<br>
MobileNetV2模型中反转残差结构和变量利用了1\*1卷积来构建最后层，以便拓展到高维的特征空间，虽然对于提取丰富特征进行预测十分重要，但却引入了比较大的计算开销和延时。为了保留高维特征的前提下减小延时，将平均池化层的层移除并用1\*1卷积来计算特征图。

### 新的激活函数h-swish
作者发现swish激活函数能够有效提高网络的精度，但是swish的计算量太大，作者提出了h-swish如下所示
![](https://img-blog.csdnimg.cn/20190513150238203.png)
![](https://img-blog.csdnimg.cn/20190513182138603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RMX3dseQ==,size_16,color_FFFFFF,t_70)

### MobileNetV3网络结构

__MobileNetV3-Large模型结构__
![](https://img-blog.csdnimg.cn/20190513182827757.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RMX3dseQ==,size_16,color_FFFFFF,t_70)
<br>
__MobileNetV3-Small模型结构__
![](https://img-blog.csdnimg.cn/20190925190912276.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RMX3dseQ==,size_16,color_FFFFFF,t_70)
