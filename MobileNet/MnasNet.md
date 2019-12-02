__本模型诞生于基于MobileNetV2的NAS搜索结构方法，它明确地将延迟考虑进主要目标中以致于可以得到一个在精度和延迟之间达到平衡的模型__

##### MnasNet的网络结构:
![](https://img-blog.csdnimg.cn/20190628110639523.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hfX2FuZw==,size_16,color_FFFFFF,t_70)

__MnasNet本人可认为看作对于MobileNetV2的升级版，毕竟是基于MobileNetV2进行搜索得到的，作者引入了一种新颖的分解式层次搜索结构，将一个CNN模型分解为不同的块然后针对每个块搜索操作和块与块的链接关系，因此允许不同块有不同的块结构。__
