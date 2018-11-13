# 基于Tensorflow的病害猪肉预测

一．实验环境：
	实验环境是基于虚拟机VMware上安装的Ubuntn16.04,Python版本是2.7，Tensorflow版本是1.3。

二．实验步骤
	1.数据预处理：首先用numpy.load将数据读入，通过对数据进行分析，发现训练数据一共是334条，特征向量为10000维的数据。其中119条数据为label=0的数据，即健康猪肉，另外215条数据为label=1的数据，即病死猪肉。然后进行数据预处理，采用numpy.vstack和numpy.hstack将数据预处理为tensorflow框架的输入格式。结果如下：

![avatar](1.png) 
图1：预处理的特征向量 

![avatar](2.png)  
图2：预处理后标签Y 
其中，标签为[1,0]的向量标记为label=0，即健康的猪肉，标签为[0，1]的向量标记为label=1，即病死的猪肉。

2.采用Tensorflow搭建神经网络框架：通过tf.Variable设置神经网络的连接权值weights和偏置biases。通过tf.reduce_sum函数来指定损失函数为预测值和真实值的交叉熵，然后通过梯度下降优化器tf.GradientDescent来最小化交叉熵。确定学习率为0.001，训练轮次为4000轮，每轮训练8条数据。

3.实验结果：通过Tensorflow搭建的神经网络，设置每100轮输出结果，最终训练数据的准确率达到98.50%，测试数据的准确率达到85%。实验结果图如下（输出轮次较多，仅截取后10轮）：

![avatar](3.png) 
图3：训练数据准确率 

![avatar](4.png) 
图4：测试数据准确率 
	
4.实验总结：通过上述实验，可以发现对于输入数据的特征向量为10000维的数据，神经网络仍可达到较好的拟合效果。对于测试数据为的准确率仅为85%，而训练数据的准确率高达98%，说明该网络对训练数据的学习过于深入，局部点陷入过拟合，导致鲁棒性较差。后期改进可以通过对损失函数加入惩罚项，即正则化的方式，还有采用随机梯度下降方式，并且对神经网络的连接参数进行一定的修改，可以一定程度上解决这个问题。



使用numpy.load("train0.npy")同理读取数据
每条数据为行向量
train0是label为0的数据 健康猪肉
train1是label为1的数据 病死猪肉
用train0与train1进行训练
在test0a与test1a验证正确率

附件是对病死猪肉的检测数据，标签为0的样本为健康猪肉，标签为1的样本为病死猪肉。每条特征向量为矩阵中的行向量。
