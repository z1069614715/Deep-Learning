### Fast R-CNN

##### 由于R-CNN的缺点，进而引申出Fast R-CNN。

##### 1. 输入一张完整的图片
##### 2. 利用CNN卷积出feature map
##### 3. 卷积层后加入一层ROI pooling layer，将选取的物体框下取样到7*7的feature map，从而形成固定的输入
##### 4. ROI pooling layer后添加全连接层
##### 5. 最后进行softmax层分类和回归预测物体框

#### Fast R-CNN存在的问题：

##### 存在瓶颈，选择性搜索，搜索出所有的候选框。
