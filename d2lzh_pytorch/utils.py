#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
import d2lzh_pytorch as d2l
from matplotlib import pyplot as plt
from IPython import display
import torchvision
import sys
import torchvision.transforms as transforms
from torch import nn
from torch.nn import init
import time
import zipfile

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize
    
def linreg(X,w,b):
    return torch.mm(X,w)+b

def squared_loss(y_hat,y):
    return (y_hat-y.view(y_hat.size()))**2/2

def sgd(params,lr,batch_size):
    for param in params:
        param.data-=lr*param.grad/batch_size

def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j=torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j)

def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images,labels):
    use_svg_display()
    _,figs=plt.subplots(1,len(images),figsize=(12,12))
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):   
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
     
    return train_iter,test_iter

def evaluate_accuracy(data_iter,net,device=None):
    if device is None and isinstance(net,torch.nn.Module):
        device=list(net.parameters())[0].device
    acc_sum,n=0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(net,torch.nn.Module):
                net.eval() #评估模式, 这会关闭dropout
                acc_sum+=(net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().item()
                net.train() # 改回训练模式
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum+=(net(X,is_training=False).argmax(dim=1)==y).float().sum().item()
                else:
                    acc_sum+=(net(X),argmax(dim=1)==y).float().sum().item()

            n+=y.shape[0]
    return acc_sum/n

def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
             params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y).sum()  
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
                    
            l.backward()
            if optimizer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                optimizer.step()
                
            
            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
            
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc:%.3f,test acc:%.3f'
             %(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))
        
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)     
    
def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,
            legend=None,figsize=(3.5,2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals,y2_vals,linestyle=':')##log对数函数
        d2l.plt.legend(legend)
        
def corr2d(X, K):  
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net=net.to(device)
    print("training on :", device)
    loss=torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,batch_count,start=0.0,0.0,0,0,time.time()
        for X,y in train_iter:
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y).sum()  
            optimizer.zero_grad()  
            l.backward()
            optimizer.step()
            train_l_sum+=l.cpu().item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            batch_count+=1
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc:%.3f,test acc:%.3f,time %.1f sec'
             %(epoch+1,train_l_sum/batch_count,train_acc_sum/n,test_acc,time.time()-start))
        #print('batch_count %d ******** n %d'%(batch_count,n))
        ##训练的损失是计算每个周期的损失，而训练的准确率是对于每个样本中对应标签的准确率
        
def load_data_fashion_mnist(batch_size,resize=None,root='~/Datasets/FashionMNIST'):
    trans=[]  ##将trans转换为Tensor格式的resize大小,因为2步都是通过append添加的，因此想要将他们2个合并，需要用到Compose函数
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform=torchvision.transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root=root,train=True,download=False,transform=transform)
    mnist_test=torchvision.datasets.FashionMNIST(root=root,train=True,download=False,transform=transform)
    
    train_iter=torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=4)
    test_iter=torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=4)
    return train_iter,test_iter  


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:])
    
    
class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_11conv=False,stride=1):
        super(Residual,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if use_11conv:
            self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3=None
            
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        return F.relu(X+Y)
    
def load_data_jay_lyrics():
    with zipfile.ZipFile('../dataset/jaychou_lyrics.txt.zip') as  zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars=f.read().decode('utf-8')
    corpus_chars=corpus_chars.replace('\n',' ').replace('\r',' ')
    corpus_chars=corpus_chars[0:10000]
    idx_to_char=list(set(corpus_chars))  ##可以说是删除掉所有重复的字符
    char_to_idx=dict([(char,i)for i,char in enumerate(idx_to_char)])
    vocab_size=len(char_to_idx)   #总共有1027个不同的字符
    corpus_indices=[char_to_idx[char] for char in corpus_chars]#10000个字符转化为索引即整形
    return corpus_indices,char_to_idx,idx_to_char,vocab_size 
    #返回所有字符对应的索引表，不同字符对应的索引号的词典，不同索引对应的字符，所有不同的字符数
    
    
    
    
#下面的代码每次从数据里随机采样一个小批量。
def data_iter_random(corpus_indices,batch_size,num_steps,device=None):
    num_examples=(len(corpus_indices)-1)//num_steps
    epoch_size=num_examples//batch_size
    example_indices=list(range(num_examples))
    random.shuffle(example_indices)  ##存放有对应字符的随机索引
    
    def _data(pos):
        return corpus_indices[pos:pos+num_steps]
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    for i in range(epoch_size):
        i=i*batch_size
        batch_indices=example_indices[i:i+batch_size]
        X=[_data(j*num_steps) for j in batch_indices]   #相当于数据集
        Y=[_data(j*num_steps+1) for j in batch_indices] #相当于输出集
        yield torch.tensor(X,dtype=torch.float32,device=device),torch.tensor(Y,dtype=torch.float,device=device)
    ## num_steps就是每个小批量样本中的数量
    ## batch_size决定每个X中有几组小批量样本
    ## example_indices只是为了随机打乱之后取随机的开始位置
    
#下面的代码每次从数据里相邻采样一个小批量。
def data_iter_consecutive(corpus_indices,batch_size,num_steps,device=None):
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices=torch.tensor(corpus_indices,dtype=torch.float32,device=device)
    data_len=len(corpus_indices)
    batch_len=data_len//batch_size
    indices=corpus_indices[0:batch_size*batch_len].view(batch_size,batch_len)
    epoch_size=(batch_len-1)//num_steps
    for i in range(epoch_size):
        i=i*num_steps
        X=indices[:,i:i+num_steps]
        Y=indices[:,i+1:i+num_steps+1]
        yield X,Y
        
#产生bat_size个词的向量
def one_hot(x,n_class,dtype=torch.float32):
    x=x.long()
    res=torch.zeros(x.shape[0],n_class,dtype=dtype,device=x.device)
    res.scatter_(1,x.view(-1,1),1)
    return res

def to_onehot(X,n_class):
    return [one_hot(X[:,i],n_class) for i in range(X.shape[1])]

#以rnn的方式预测下一个字符
def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,
               num_hiddens,vocab_size,device,idx_to_char,char_to_idx):
    state=init_rnn_state(1,num_hiddens,device)
    output=[char_to_idx[prefix[0]]]  #首个字符对应的索引
    for t in range(num_chars+len(prefix)-1):
        X=to_onehot(torch.tensor([[output[-1]]],device=device),vocab_size)
        (Y,state)=rnn(X,state,params)
        if t<len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

#假如梯度本身就很小了再缩小不是更会出现梯度衰减的情况么??
def grad_clipping(params,theta,device):
    norm=torch.tensor([0.0],device=device)
    for param in params:
        norm+=(param.grad.data**2).sum()
    norm=norm.sqrt().item()
    if norm>theta:
        for param in params:
            param.grad.data*=(theta/norm)

            
#模型训练函数
def train_and_predict_rnn(rnn,get_params,init_rnn_state,num_hiddens,
                         vocab_size,device,corpus_indices,idx_to_char,
                         char_to_idx,is_random_iter,num_epochs,num_steps,
                         lr,clipping_theta,batch_size,pred_period,pred_len,prefixes):
    if is_random_iter:
        data_iter_fn=d2l.data_iter_random
    else:
        data_iter_fn=d2l.data_iter_consecutive
    params=get_params()
    loss=nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        if not is_random_iter:
            state=init_rnn_state(batch_size,num_hiddens,device)
        l_sum,n,start=0.0,0,time.time()
        data_iter=data_iter_fn(corpus_indices,batch_size,num_steps,device)
        for X,Y in data_iter:
            if is_random_iter:
                state=init_rnn_state(batch_size,num_hiddens,device)
            else:
                for s in state:
                    s.detach_()
                    
            inputs=to_onehot(X,vocab_size)
            (outputs,state)=rnn(inputs,state,params)
            outputs=torch.cat(outputs,dim=0)
            y=torch.transpose(Y,0,1).contiguous().view(-1)
            l=loss(outputs,y.long())
            
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params,clipping_theta,device)
            d2l.sgd(params,lr,1)
            l_sum+=l.item()*y.shape[0]   
            n+=y.shape[0] ##因为我这里要求的是所有epoch累计的损失函数的平均值
            
        if (epoch+1) % pred_period==0:
            print('epoch %d,perplexity %f,time %.2f sec'%
                   (epoch+1,math.exp(l_sum/n),time.time()-start))
            for prefix in prefixes:
                print(' -',predict_rnn(prefix,pred_len,rnn,params,
                                       init_rnn_state,num_hiddens,vocab_size,
                                     device,idx_to_char,char_to_idx))
                
                
                
class RNNModel(nn.Module):
    def __init__(self,rnn_layer,vocab_size):
        super(RNNModel,self).__init__()
        self.rnn=rnn_layer
        self.hidden_size=rnn_layer.hidden_size*(2 if rnn_layer.bidirectional else 1)
        self.vocab_size=vocab_size
        self.dense=nn.Linear(self.hidden_size,vocab_size)
        self.state=None
    def forward(self,inputs,state):
        X=d2l.to_onehot(inputs,self.vocab_size)
        Y,self.state=self.rnn(torch.stack(X),state)
        outputs=self.dense(Y.view(-1,Y.shape[-1]))
        return outputs,self.state ##相当于获得每个词典的得分
    
    
#以RNN的方式预测
def predict_rnn_pytorch(prefix,num_chars,model,vocab_size,device,idx_to_char,
                       char_to_idx):
    state=None
    output=[char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        X=torch.tensor([output[-1]],device=device).view(1,1)
        if state is not None:
            if isinstance(state,tuple):
                state=(state[0].to(device),state[1].to(device))
            else:
                state=state.to(device)
                
        (Y,state)=model(X,state)
        if(t<len(prefix)-1):
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

#训练以及预测歌词
def train_and_predict_rnn_pytorch(model,num_hiddens,vocab_size,device,
                                 corpus_indices,idx_to_char,char_to_idx,
                                 num_epochs,num_steps,lr,clipping_theta,
                                 batch_size,pred_period,pred_len,prefixes):
    #pred_period每多少周期打印一次
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    model.to(device)
    state=None
    for epoch in range(num_epochs):
        l_sum,n,start=0.0,0,time.time()
        data_iter=d2l.data_iter_consecutive(corpus_indices,batch_size,num_steps,device)
        for X,Y in data_iter:
            if state is not None:
                if isinstance(state,tuple):
                    state=(state[0].detach(),state[1].detach())
                else:
                    state=state.detach()
                
            (output,state)= model(X,state)
            ##我总感觉是transpose(Y,1,0),试验了困惑度还小一些,想法正确！！！
            y=torch.transpose(Y,0,1).contiguous().view(-1)
            l=loss(output,y.long())
            
            optimizer.zero_grad()
            l.backward()
            d2l.grad_clipping(model.parameters(),clipping_theta,device)
            optimizer.step()
            l_sum+=l.item()*y.shape[0]
            n+=y.shape[0]
            
        try:
            perplexity=math.exp(l_sum/n)
        except OverflowError:
            perplexity=float('inf')
        if(epoch+1)%pred_period==0:
            print('epoch %d,perplexity %f,time %.2f sec'%(
            epoch+1,perplexity,time.time()-start))
            for prefix in prefixes:
                print('-',predict_rnn_pytorch(
                prefix,pred_len,model,vocab_size,device,idx_to_char,
                char_to_idx))

#以2维的方式画出3d中梯度下降的形式。      
def train_2d(trainer):
    x1,x2,s1,s2=-5,-2,0,0
    results=([(x1,x2)])
    for i in range(20):
        x1,x2,s1,s2=trainer(x1,x2,s1,s2)
        results.append((x1,x2))
    print('epoch %d,x1 %f,x2 %f'%(i+1,x1,x2))
    return results

def show_trace_2d(f,results):
    d2l.plt.plot(*zip(*results),'-o',color='#ff7f0e')
    x1,x2=np.meshgrid(np.arange(-5.5,1.0,0.1),np.arange(-3.0,1.0,0.1))
    d2l.plt.contour(x1,x2,f(x1,x2),colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    
#故意的1500个飞机机翼噪音的数据集
def get_data_ch7():
    data=np.genfromtxt('../dataset/airfoil_self_noise.dat',delimiter='\t')
    data=(data-data.mean(axis=0))/data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
    torch.tensor(data[:1500, -1], dtype=torch.float32)    


#飞机机翼噪音的梯度训练
def train_ch7(optimizer_fn,states,hyperparams,features,labels,
             batch_size=10,num_epochs=2):
    net,loss=d2l.linreg,d2l.squared_loss
    w=torch.nn.Parameter(torch.tensor(np.random.normal(0,0.01,size=(features.shape[1],1)),
                                      dtype=torch.float32),requires_grad=True)
    b=torch.nn.Parameter(torch.zeros(1,dtype=torch.float32),requires_grad=True)
    
    def eval_loss():
        return loss(net(features,w,b),labels).mean().item()
    ls=[eval_loss()]
    
    data_iter=torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(features,labels),batch_size,shuffle=True)
    
    for _ in range(num_epochs):
        start=time.time()
        for batch_i,(X,y) in enumerate(data_iter):
            l=loss(net(X,w,b),y).mean()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            optimizer_fn([w,b],states,hyperparams)
            if (batch_i+1)*batch_size%100==0:
                ls.append(eval_loss())
                
    print('loss:%f,%f sec per epoch'%(ls[-1],time.time()-start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0,num_epochs,len(ls)),ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    
def train_pytorch_ch7(optimizer_fn,optimizer_hyperparams,features,labels,
                     batch_size=10,num_epochs=2):
    net=nn.Sequential(
        nn.Linear(features.shape[-1],1)
    )
    loss=nn.MSELoss()
    optimizer=optimizer_fn(net.parameters(),**optimizer_hyperparams)
    
    def eval_loss():
        return loss(net(features).view(-1),labels).item()/2
    
    ls=[eval_loss()]
    data_iter=torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(features,labels),batch_size,shuffle=True)
    
    for _ in range(num_epochs):
        start=time.time()
        for batch_i,(X,y) in enumerate(data_iter):
            l=loss(net(X).view(-1),y)/2
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i+1)*batch_size%100==0:
                ls.append(eval_loss())
    print('loss:%f,%f sec per'%(ls[-1],time.time()-start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0,num_epochs,len(ls)),ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    
#计时器
class Benchmark():  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))

#显示图像
def show_images(imgs,num_rows,num_cols,scale=2):
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_11conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

def resnet18(output=10, in_channels=3):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, output))) 
    return net

#图像增广的训练模型
def train(train_iter,test_iter,net,loss,optimizer,device,num_epochs):
    net=net.to(device)
    print("traing on ",device)
    batch_count=0
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start=0.0,0.0,0,time.time()
        for X,y in train_iter:
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum+=l.cpu().item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            batch_count+=1
        test_acc=d2l.evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f,time %.1f sec'
             %(epoch+1,train_l_sum/batch_count,train_acc_sum/n,test_acc,time.time()-start))
        
        
        
        
#将左上左下坐标转化为matplotlib的边界框格式，即（X,Y,宽，高）
def bbox_to_rect(bbox,color):
    return d2l.plt.Rectangle(
    xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],
    fill=False,edgecolor=color,linewidth=2)


#生成一张图的锚框的归一化后的坐标
def MultiBoxPrior(feature_map,sizes=[0.75,0.5,0.25],ratios=[1,2,0.5]):
    pairs=[]
    for r in ratios:
        pairs.append([sizes[0],math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s,math.sqrt(ratios[0])])
        
    pairs=np.array(pairs)
    
    ss1=pairs[:,0]*pairs[:,1]
    ss2=pairs[:,0]/pairs[:,1]
    base_anchors=np.stack([-ss1,-ss2,ss1,ss2],axis=1)/2
    
    h,w=feature_map.shape[-2:]
    shifts_x=np.arange(0,w)/w
    shifts_y=np.arange(0,h)/h
    shift_x,shift_y=np.meshgrid(shifts_x,shifts_y)
    shift_x,shift_y=shift_x.reshape(-1),shift_y.reshape(-1)
    shifts=np.stack((shift_x,shift_y,shift_x,shift_y),axis=1)
    
    anchors=shifts.reshape(-1,1,4)+base_anchors.reshape((1,-1,4))
    return torch.tensor(anchors,dtype=torch.float32).view(1,-1,4)


#生成一张图的锚框
def show_bboxes(axes,bboxes,labels=None,colors=None):
    def _make_list(obj,default_values=None):
        if obj is None:
            obj=default_values
        elif not isinstance(obj,(list,tuple)):
            obj=[obj]
        return obj
    
    labels=_make_list(labels)
    colors=_make_list(colors,['b','g','r','m','c'])
    for i,bbox in enumerate(bboxes):
        color=colors[i%len(colors)]
        rect=d2l.bbox_to_rect(bbox.detach().cpu().numpy(),color)
        axes.add_patch(rect)
        if labels and len(labels)>i:
            text_color='k' if color=='w' else 'w'
            axes.text(rect.xy[0],rect.xy[1],labels[i],va='center',
                     ha='center',fontsize=6,color=text_color,
                     bbox=dict(facecolor=color,lw=0))
            
            
            
##计算交并比
def compute_intersection(set_1,set_2):
    low_bounds=torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unqueeze(0))
    upper_bounds=torch.min(set_1[:,2:].unsqueeze(1),set_2[:,2:].unsqueeze(0))
    intersection_dims=torch.clamp(upper_bounds-low_bounds,min=0)
    return intersection_dims[:,:,0]*intersection_dims[:,:,1]

def compute_jaccard(set_1,set_2):
    intersection=compute_intersection(set_1,set_2)
    areas_set_1=(set_1[:,2]-set_1[:,0])*(set_1[:,3]-set_1[:,1])
    areas_set_2=(set_2[:,2]-set_2[:,0])*(set_2[:,3]-set_2[:,1])
    union=areas_set_1.unsqueeze(1)+areas_set_2.unsqueeze(0)-intersection
    return intersection/union  #（交 /并）


##预测锚框的偏差值和标签
def assign_anchor(bb,anchor,jaccard_threshold=0.5):
    na=anchor.shape[0]
    nb=bb.shape[0]
    jaccard=compute_jaccard(anchor,bb).detach().cpu().numpy()
    assigned_idx=np.ones(na)*-1
    jaccard_cp=jaccard.copy()
    for j in range(nb):
        i=np.argmax(jaccard_cp[:,j])
        assigned_idx[i]=j
        jaccard_cp[i,:]=float("-inf")
        
    for i in range(na):
        if assigned_idx[i]==-1:
            j=np.argmax(jaccard[i,:])
            if jaccard[i,j]>=jaccard_threshold:
                assigned_idx[i]=j
    return torch.tensor(assigned_idx,dtype=torch.long)

def xy_to_cxcy(xy):
    return torch.cat([(xy[:,2:]+xy[:,:2])/2,
                     xy[:,2:]-xy[:,:2]],1)

def MultiBoxTarget(anchor,label):
    assert len(anchor.shape)==3 and len(label.shape)==3
    bn=label.shape[0]
    
    #每次处理一个标签
    def MultiBoxTarget_one(anc,lab,eps=1e-6):
        an=anc.shape[0]
        assigned_idx=assign_anchor(lab[:,1:],anc)
        bbox_mask=((assigned_idx>=0).float().unsqueeze(-1)).repeat(1,4)
        cls_labels=torch.zeros(an,dtype=torch.long)
        assigned_bb=torch.zeros((an),4,dtype=torch.float32)
        for i in range(an):
            bb_idx=assigned_idx[i]
            if bb_idx>=0:
                cls_labels[i]=lab[bb_idx,0].long().item()+1
                assigned_bb[i,:]=lab[bb_idx,1:]
                
        center_anc=xy_to_cxcy(anc)
        center_assigned_bb=xy_to_cxcy(assigned_bb)
        
        ##偏移量,
        offset_xy=10.0*(center_assigned_bb[:,:2]-center_anc[:,:2])/center_anc[:,:2]#标准代码这里写错了
        offset_wh=5.0*(center_assigned_bb[:,2:]-center_anc[:,2:])/center_anc[:,2:]
        offset=torch.cat([offset_xy,offset_wh],dim=1)*bbox_mask
        return offset.view(-1),bbox_mask.view(-1),cls_labels
    
    batch_offset=[]
    batch_mask=[]
    batch_cls_labels=[]
    for b in range(bn):
        offset,bbox_mask,cls_labels=MultiBoxTarget_one(anchor[0,:,:],label[b,:,:])
        
        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)
    bbox_offset=torch.stack(batch_offset)
    bbox_mask=torch.stack(batch_mask)
    cls_labels=torch.stack(batch_cls_labels)
    
    return [bbox_offset,bbox_mask,cls_labels]


#非极大值抑制然后输出
# 以下函数已保存在d2lzh_pytorch包中方便以后使用
from collections import namedtuple
Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])

def non_max_suppression(bb_info_list, nms_threshold = 0.5):
    """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
    """
    output = []
    # 先根据置信度从高到低排序
    sorted_bb_info_list = sorted(bb_info_list, key = lambda x: x.confidence, reverse=True)

    while len(sorted_bb_info_list) != 0:
        best = sorted_bb_info_list.pop(0)
        output.append(best)

        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)

        iou = compute_jaccard(torch.tensor([best.xyxy]), 
                              torch.tensor(bb_xyxy))[0] # shape: (len(sorted_bb_info_list), )

        n = len(sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]
    return output

def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold = 0.5):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]

    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold = 0.5):
        """
        MultiBoxDetection的辅助函数, 处理batch中的一个
        Args:
            c_p: (预测总类别数+1, 锚框个数)
            l_p: (锚框个数*4, )
            anc: (锚框个数, 4)
            nms_threshold: 非极大抑制中的阈值
        Return:
            output: (锚框个数, 6)
        """
        pred_bb_num = c_p.shape[1]
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy() # 加上偏移量

        confidence, class_id = torch.max(c_p, 0)
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()

        pred_bb_info = [Pred_BB_Info(
                            index = i,
                            class_id = class_id[i] - 1, # 正类label从0开始
                            confidence = confidence[i],
                            xyxy=[*anc[i]]) # xyxy是个列表
                        for i in range(pred_bb_num)]

        # 正类的index
        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]

        output = []
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])

        return torch.tensor(output) # shape: (锚框个数, 6)

    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))

    return torch.stack(batch_output)

#读取皮卡丘的数据集
class PikachuDetDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,part,image_size=(256,256)):
        assert part in ["train","val"]
        self.image_size=image_size
        self.image_dir=os.path.join(data_iter,part,"images")
        with open(os.path.join(data_iter,part,"label.json")) as f:
            self.label=json.load(f)
        self.transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        image_path=str(index+1)+".png"
        print(index,'**********')
        cls=self.label[image_path]["class"]
        label=np.array([cls]+self.label[image_path]["loc"],dtype="float32")[None,:]
        
        PIL_img = Image.open(os.path.join(self.image_dir, image_path)
                            ).convert('RGB').resize(self.image_size)
        img=self.transform(PIL_img)
        
        sample={
            "label":label,
            "image":img
        }
        return sample
    
def load_data_pikachu(batch_size,edge_size=256,data_dir='../dataset/pikachu'):
    image_size=(edge_size,edge_size)
    train_dataset=PikachuDetDataset(data_dir,'train',image_size)
    val_dataset=PikachuDetDataset(data_dir,'val',image_size)
    train_iter=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                                          shuffle=True,num_workers=0)
    val_iter=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,
                                        shuffle=True,num_workers=0)
    return train_iter,val_iter
