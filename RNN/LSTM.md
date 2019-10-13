##### LSTM => `long short term memory` 是为了解决RNN中记录长期记忆效果不佳而提出

以下是RNN的网络架构:<br>
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDVCVnl6bWx5MWZvdG41Y3l6eXBqMzBqZzA3YTc0eC5qcGc?x-oss-process=image/format,png)
<br>
LSTM 同样是RNN的结构，但是重复的模块拥有一个不同的结构:<br>
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDVCVnl6bWx5MWZvdG5hdHhzbTdqMzBqZzA3YmpzaS5qcGc?x-oss-process=image/format,png)

##### LSTM中通过一种叫`门(gates)`的结构来实现
##### 门可以选择性地让信息通过， 主要通过一个sigmoid的神经层和一个逐点相乘的操作来实现<br>

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDVCVnl6bWx5MWZvdG54dHFqbG1qMzA1aTA2cTN5ZC5qcGc?x-oss-process=image/format,png)

##### 问题来了，为什么要通过sigmoid层来当激活函数？
因为sigmoid层输出范围是0~1，就相当于表示对应信息通过的权重，`0`==>对应信息不让通过`权重为0`，`1`==>对应信息全部通过`权重为1`

##### LSTM通过这么三个这样的结构来实现。`输入们，遗忘门，输出门`

* LSTM第一步是决定我们需要遗忘什么信息
该门会读取`h(t-1)`(t-1层的输出)和`x(t)`(t层的输入)，通过一个sigmoid函数来输出一个0~1的值来表示`c(t-1)`t-1层中细胞的状态，1表示完全保留，0表示完成舍弃。<br>
##### 这个门主要工作是得到上一个细胞状态需要保留的权重，决定保留多少的信息
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDVCVnl6bWx5MWZvdG9hOXNnbTVqMzBqZzA2MDc0cS5qcGc?x-oss-process=image/format,png)
<br>例如：细胞状态可能包含当前主语的性别，因此正确的代词可以被选择出来。当我们看到新的主语，我们希望忘记旧的主语。

* LSTM第二步是决定让多少新信息加入到细胞状态中
##### 这个门主要工作是得出新的候选信息的权重，决定需要多少的新信息，然后结合遗忘门来得到这层细胞的状态
通过一个sigmoid层来得到`i(t)`表示需要更新信息的权重，然后一个tanh层来得到更新的信息，将这两个部分联合，对cell状态的一个更新。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDVCVnl6bWx5MWZvdG9mdHc1MGlqMzFlcTBmb216NC5qcGc?x-oss-process=image/format,png)
下列公式就是对应本层cell的更新状态
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDVCVnl6bWx5MWZvdG9qYWI1bWhqMzBqZzA2MHdlei5qcGc?x-oss-process=image/format,png)
<br>例如：在语言模型的例子中，这就是我们实际根据前面确定的目标，丢弃旧代词的性别信息并添加新的信息的地方。

* LSTM第三步是决定本层的输出
##### 这个门的主要工作就是决定本层的输出`h(t)`
通过一层sigmoid来决定将细胞状态中的那个部分输出出去，最后把细胞状态通过一层tanh来进行处理，并将它和sigmoid层的输出相乘。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDVCVnl6bWx5MWZvdG9reTI4emJqMzBqZzA2MDc0dy5qcGc?x-oss-process=image/format,png)
<br>例如：在语言模型的例子中，因为他就看到了一个 代词，可能需要输出与一个 动词 相关的信息。例如，可能输出是否代词是单数还是负数，这样如果是动词的话，我们也知道动词需要进行的词形变化。

