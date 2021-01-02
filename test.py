# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:03:01 2020
@author: SUE
"""
 
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def Leaky_Relu(x):
    alpha=1
    r=np.where(x<0)
    x[r]=x[r]*alpha
    return x    
def tansig(x):
    return np.tanh(x)
def softmax(x):
    exp_x= np.exp(x)
    exp_x=exp_x/sum(exp_x)
    return exp_x 
def diff_sigmoid(x):
    return np.exp(-x)/(1+np.exp(-x))/(1+np.exp(-x))
def diff_Leaky_Relu(x):
    alpha=1
    DX=x
    p=np.where(x<0)
    q=np.where(x>0)
    DX[p]=alpha
    DX[q]=1
    return DX
def diff_tansig(x):
    return 1-(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))*(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def diff_softmax(x):
    soft=softmax(x)
    return(soft)*(np.ones(soft.shape)-soft)
def process(W,b,x,Act_fun):
    return Act_fun(np.dot(W,x)+b)
def qua_Err(x,OUTPUT):
    return 2*(x-OUTPUT.T)
def cross_entropy_Err(x,OUTPUT):#求完
    return OUTPUT.T*np.log(x)
def Data_import():
    filename='mnist_train.csv'
    with open(filename,'rt') as raw_data:
        data=np.loadtxt(raw_data,delimiter=',')
        INPUT=data[:,1:785]
        OUTPUT=data[:,0]
        print("数据导入结束")
    return INPUT,OUTPUT
def Normalize(INPUT,OUTPUT):#采用minmax归一化
    IN=INPUT/255
    OUT=np.zeros((10,len(OUTPUT)))
    for i in range(len(OUTPUT)):
        OUT[int(OUTPUT[i])][i]=1
    print("数据归一化结束")
    return IN,OUT
def set_up_BP(NeuNumList):
        W=[]#连接阵
        b=[]#偏置
        for i in range(len(NeuNumList)):
            if(i<len(NeuNumList)-1):
                W.append(np.random.randn(NeuNumList[i+1],NeuNumList[i]))
                b.append(np.random.randn(NeuNumList[i+1],1))
        return W,b
def get_first_x(W,b,Act,INPUT):#输入参数Act是激活函数指针
    x=[]
    x.append(INPUT)
    for i in range(len(b)):
       # print(x[len(x)-1].shape,W[i].shape)
        x.append(process(W[i],b[i],x[len(x)-1],Act[i]))
    return x 
def get_diff_list(W,b,Act,x,OUTPUT,ERRFUN):#最后一列是损失函数类型
    #返回的是最终误差对于每个参数的偏导数
    ERR=ERRFUN(x[len(x)-1],OUTPUT)
  #  print(ERR.shape)          
    DW=[]
    Db=[]
    for i in range(len(b)): 
         TW=[]
         Tb=[]
         D=[]
         if(Act[len(b)-1-i]==sigmoid):
                D=x[len(b)-i]*(1-x[len(b)-i])
         elif(Act[len(b)-1-i]==tansig):
                D=1-x[len(b)-i]*x[len(b)-i]
         elif(Act[len(b)-1-i]==softmax):
                D=x[len(b)-i]*(1-x[len(b)-i])
         Tb=ERR*D
        # print(x[len(b)-1-i].shape)
         TW=np.dot(Tb,x[len(b)-1-i].T)
        # ERR=[]
         ERR=np.dot(W[len(b)-1-i].T,ERR)
         DW.append(TW)
       #  print(sum(Tb.T).shape)
         Db.append(sum(Tb.T))
         #print(Db)
    return DW,Db
def Mini_batch(length,INPUT,OUTPUT):#将数据集分隔
    num=240#将数据集分为100份
    Combination=np.array(range(0,length))
    np.random.shuffle(Combination)
    Combination=Combination.reshape(num,int(length/num))   
    print("数据分割结束")
    return INPUT[Combination],OUTPUT[Combination]
def Drill(W,b,Act,INPUT,OUTPUT,ERRFUN,epoch,step=0.01,show_epoch=1):#最后两个元素是训练步长和多少次展示下当前MSE
    #lasts=100000
    OUTPUT=OUTPUT.T
    INPUT,OUTPUT=Mini_batch(len(OUTPUT), INPUT, OUTPUT)
    print(INPUT.shape,OUTPUT.shape)
    num=len(INPUT)
    s=0
    beta=0.1#动量占比
    PW=[]
    Pb=[]
    for j in range(len(b)):
               PW.append(np.zeros(W[len(b)-1-j].shape))
               Pb.append(np.zeros((len(b[len(b)-1-j]),1)))
    for i in range(epoch):
        if i%num==0:
           print("正确率:%f"%(s/num))
           s=0
        Db=[]
        DW=[]
        x=[]
        x=get_first_x(W,b,Act,INPUT[i%num].T)
        s=s+validate(x[len(x)-1].T,OUTPUT[i%num])
        DW,Db=get_diff_list(W,b,Act,x,OUTPUT[i%num],ERRFUN)
        for j in range(len(b)):#收集梯度
            W[j]=W[j]-step*PW[len(b)-1-j]
            b[j]=b[j]-step*Pb[len(b)-1-j].reshape(-1,1)
            PW[j]=(1-beta)*PW[j]+beta*DW[j]
            Pb[j]=(1-beta)*Pb[j]+beta*Db[j].reshape(-1,1)
    return W,b
def validate(x,OUTPUT):#计算正确率
    v=np.argmax(x,1)
    h=np.argmax(OUTPUT,1)
    r=np.where(v-h==0)
    return len(r[0])/len(h)
def predict(W,b,Act,INPUT):
    x=np.array(INPUT,ndmin=2).reshape(1,-1)
    for i in range(len(b)):
        x=process(W[i],b[i],x,Act[i])
    return x 
 
if __name__ == "__main__":
        W,b=set_up_BP([784,200,10])
        Act=[tansig,sigmoid]
        INPUT,OUTPUT=Data_import()
        Iter=30000
        IN,OUT=Normalize(INPUT,OUTPUT)
        W,b=Drill(W,b,Act,IN,OUT,qua_Err,Iter,0.0001,1)
        #检测测试集上的正确率:
        del IN,OUT,INPUT,OUTPUT
        filename='mnist_test.csv'
        with open(filename,'rt') as raw_data:
            data=np.loadtxt(raw_data,delimiter=',')
            INPUT=data[:,1:785]
            OUTPUT=data[:,0]
            IN,OUT=Normalize(INPUT,OUTPUT)
            del INPUT,OUTPUT
        x=np.array(IN,ndmin=2).T
        for i in range(len(b)):
             x=process(W[i],b[i],x,Act[i])
        print("测试集上的正确率是:%f"%(validate(x.T,OUT.T)))