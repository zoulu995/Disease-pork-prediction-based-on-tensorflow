# coding:utf-8
import numpy as np
data = np.load("train0.npy")  # m=119, n=10000
data1 = np.load("train1.npy") # m=215, n=10000
data = np.vstack([data,data1])
m, n = np.shape(data)
# print "m=",m
# print "n=",n
# print(data)
# print type(data[0][0]) # <type 'numpy.float64'>

y1 = np.zeros((119,1))
y1temp = np.ones((119,1))
y1 = np.hstack((y1temp,y1))
y2 = np.ones((215,1))
y2temp = np.zeros((215,1))
y2 = np.hstack((y2temp,y2))
y = np.vstack((y1,y2))
print "y:\n",y
m , n = np.shape(y)
print "y's m is:",m
print "y's n is:",n



testdata0 = np.load("test0a.npy")  # m1= 20,n1= 10000
testdata1 = np.load("test1a.npy") # m1= 20,n1= 10000
testdata = np.vstack((testdata0,testdata1))
m1 , n1 = np.shape(testdata)
print "m1=",m1
print "n1=",n1
print "**************"
yt1 = np.zeros((20,1))
yt1temp = np.ones((20,1))
yt1 = np.hstack((yt1temp,yt1))
yt2 = np.ones((20,1))
yt2temp = np.zeros((20,1))
yt2 = np.hstack((yt2temp,yt2))
yt = np.vstack((yt1,yt2))
print yt

ytest = np.zeros((20,1))
ytest = np.hstack((ytest,yt2))
# print "ytest:\n"
# print ytest
