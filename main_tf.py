# coding=utf-8

import tensorflow as tf
import numpy as np

# 训练数据
data = np.load("train0.npy")  # m=119, n=10000
data1 = np.load("train1.npy") # m=215, n=10000
X = np.vstack([data,data1])

y1 = np.zeros((119,1))
y1temp = np.ones((119,1))
y1 = np.hstack((y1temp,y1))
y2 = np.ones((215,1))
y2temp = np.zeros((215,1))
y2 = np.hstack((y2temp,y2))
Y = np.vstack((y1,y2))

# 测试数据
testdata0 = np.load("test0a.npy")  # m1= 20,n1= 10000
testdata1 = np.load("test1a.npy") # m1= 20,n1= 10000
xt = np.vstack((testdata0,testdata1))

yt1 = np.zeros((20,1))
yt1temp = np.ones((20,1))
yt1 = np.hstack((yt1temp,yt1))
yt2 = np.ones((20,1))
yt2temp = np.zeros((20,1))
yt2 = np.hstack((yt2temp,yt2))
yt = np.vstack((yt1,yt2))

# 设置权重weights和偏置biases作为优化变量，初始值设为0
weights = tf.Variable(tf.zeros([10000, 2]))
biases = tf.Variable(tf.zeros([2]))

# 构建神经网络模型
x = tf.placeholder("float", [None, 10000])
y = tf.nn.softmax(tf.matmul(x, weights) + biases)# 模型的预测值
y_real = tf.placeholder("float", [None, 2])# 真实值

# 预测值与真实值的交叉熵
cross_entropy = -tf.reduce_sum(y_real * tf.log(y))
# 使用梯度下降优化器最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


print "X:\n",X
print "Y:\n",Y
m, n = np.shape(X)
print "m=",m
print "n=",n


# 开始训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
BATCH_SIZE = 8
for i in range(4000):
	start = (i*BATCH_SIZE) % 334
	end = start + BATCH_SIZE
	sess.run(train_step, feed_dict={x: X[start:end], y_real:Y[start:end]})
	if i % 100 == 0:
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_real, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print "准确率为:",
		print sess.run(accuracy, feed_dict={x:xt, y_real:yt})
