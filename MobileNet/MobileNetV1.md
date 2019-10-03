### MobileNetV1

##### 特点:
* 减少参数数量和计算量的原理。深度可分离卷积。<br>
![](https://img-blog.csdn.net/20181010155505157)
* 用 `CONV/s2`（步进2的卷积）代替 `MaxPool+CONV`：使得参数数量不变，计算量变为原来的 1/4 左右，且省去了MaxPool的计算量。
* 采用 depth-wise convolution 会有一个问题，就是导致 信息流通不畅 ，即输出的 feature map 仅包含输入的 feature map 的一部分，在这里，MobileNet 采用了 point-wise(1*1) convolution 帮助信息在通道之间流通

##### MobileNetV1中引入的两个超参数:
* Width Multiplier(α \alphaα): Thinner Models
  * 所有层的 通道数（channel） 乘以 α 参数(四舍五入)，模型大小近似下降到原来的 α2 倍，计算量下降到原来的 α2 倍。
  * α∈(0,1] with typical settings of 1, 0.75, 0.5 and 0.25，降低模型的宽度。
* Resolution Multiplier(ρ \rhoρ): Reduced Representation
  * 输入层的 分辨率（resolution） 乘以 ρ 参数 (四舍五入)，等价于所有层的分辨率乘 ρ ，模型大小不变，计算量下降到原来的 ρ2 倍。
  * ρ∈(0,1]，降低输入图像的分辨率。
 
 ##### 标准卷积和深度可分离卷积的区别：
 ![](https://img-blog.csdn.net/20180501154941666)
 
 ##### caffe中的代码实现:
 ![](https://img-blog.csdnimg.cn/20190802165214128.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L216cG16aw==,size_16,color_FFFFFF,t_70)
