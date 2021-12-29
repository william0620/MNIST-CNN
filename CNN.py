import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
import os

#调用GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("You are using",device)
torch.cuda.empty_cache()

#初始化变量
n_epochs = 5 #训练次数
batch_size_train = 240 #训练的 batch_size
batch_size_test = 1000 #测试的 batch_size
learning_rate = 0.001 # 学习率
momentum = 0.5 # 在梯度下降过程中解决mini-batch SGD优化算法更新幅度摆动大的问题，使得收敛速度更快
log_interval = 10 # 操作间隔
random_seed = 2 # 随机种子，设置后可以得到稳定的随机数
torch.manual_seed(random_seed)

#导入训练集并增强数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.RandomAffine(degrees = 0,translate=(0.1, 0.1)),
                                   torchvision.transforms.RandomRotation((-10,10)),#将图片随机旋转（-10,10）度
                                   torchvision.transforms.ToTensor(),# 将PIL图片或者numpy.ndarray转成Tensor类型
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                              ),
    batch_size=batch_size_train, shuffle=True,pin_memory=True) # shuffle如果为true,每个训练epoch后，会将数据顺序打乱

#导入测试集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                              ),
    batch_size=batch_size_test, shuffle=True, pin_memory=True)

# 用 enumerate 加载测试集
examples = enumerate(test_loader)
# 获取一个 batch
batch_idx, (example_data, example_targets) = next(examples)
# 查看 batch 数据，有10000张图像的标签，tensor 大小为 [1000, 1, 28, 28]
# 即图像为 28 * 28， 1个颜色通道（灰度图）， 1000张图像
#print(example_targets)
#print(example_data.shape)

#查看部分图片
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)# 创建 subplot
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Label: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels =32 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_drop = nn.Dropout(0.25)

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0 )
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0 )
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_drop = nn.Dropout(0.25)

        # Fully-Connected layer 1
        
        self.fc1 = nn.Linear(576,256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        
        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,10)
                
    def forward(self, x):
        # conv layer 1 的前向计算
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)
        
        out = self.maxpool1(out)
        out = self.conv1_drop(out)

        # conv layer 2 的前向计算
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)
        
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.batch4(out)
        
        out = self.maxpool2(out)
        out = self.conv2_drop(out)

        #Flatten拉平操作
        out = out.view(out.size(0),-1)

        #FC layer的前向计算
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dp1(out)
        
        out = self.fc2(out)

        return F.log_softmax(out,dim = 1)

#权值初始化
def weight_init(m):
    # 根据网络层的不同定义不同的初始化方式,使用吴恩达推荐的relu初始化方式 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    '''
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    ''' 

# 实例化一个网络
network = CNNModel()
network.to(device)
#调用权值初始化函数
network.apply(weight_init)
# 设置优化器，用stochastic gradient descent，设置学习率，设置momentum
#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = optim.Adam(network.parameters(), lr=learning_rate)
optimizer = optim.RMSprop(network.parameters(),lr=learning_rate,alpha=0.99,momentum = momentum)
#设置学习率梯度下降，如果连续三个epoch测试准确率没有上升，则降低学习率
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

#定义存储数据的列表
train_losses = []
train_counter = []
train_acces = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
test_acces = []

# 定义训练函数
def train(epoch):
    
    network.train() # 将网络设为 training 模式
    train_correct = 0
    # 对一组 batch 
    for batch_idx, (data, target) in enumerate(train_loader): 
        # 通过enumerate获取batch_id, data, and label
        # 1-将梯度归零
        optimizer.zero_grad()
        
        # 2-传入一个batch的图像，并前向计算
        # data.to(device)把图片放入GPU中计算
        output = network(data.to(device))
        
        # 3-计算损失
        loss = F.nll_loss(output, target.to(device))
        
        # 4-反向传播
        loss.backward()
        
        # 5-优化参数
        optimizer.step()
        #exp_lr_scheduler.step()
        
        train_pred = output.data.max(dim=1, keepdim=True)[1] # 取 output 里最大的那个类别, 
             # dim = 1表示去每行的最大值，[1]表示取最大值的index，而不去最大值本身[0]    

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum() # 比较并求正确分类的个数
        #打印以下信息：第几个epoch，第几张图像， 总训练图像数, 完成百分比，目前的loss
        print('\r 第 {} 次 Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),end = '')

        # 每第10个batch (log_interval = 10)
        if batch_idx  % log_interval == 0:
            #print(batch_idx)
            # 把目前的 loss加入到 train_losses,后期画图用
            train_losses.append(loss.item())
            # 计数
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    
    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))

# 定义测试函数
def test(epoch):
    network.eval() # 将网络设为 evaluating 模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device)) # 传入这一组 batch，进行前向计算
            #test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()

            pred = output.data.max(dim=1, keepdim=True)[1] # 取 output 里最大的那个类别, 
             # dim = 1表示去每行的最大值，[1]表示取最大值的index，而不去最大值本身[0]    

            correct += pred.eq(target.data.view_as(pred).to(device)).sum() # 比较并求正确分类的个数
    acc = correct / len(test_loader.dataset)# 平均测试准确率
    test_acces.append(acc.cpu().numpy().tolist())
    
    test_loss /= len(test_loader.dataset) # 平均 loss， len 为 10000
    test_losses.append(test_loss) # 记录该 epoch 下的 test_loss
    
    #保存测试准确率最大的模型
    if test_acces[-1] >= max(test_acces):
        # 每个batch训练完后保存模型 
        torch.save(network.state_dict(), './model02.pth')

        # 每个batch训练完后保存优化器
        torch.save(optimizer.state_dict(), './optimizer02.pth')
    
    # 打印数据训练相关信息
    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n'\
          .format(epoch,test_loss, correct,len(test_loader.dataset),100. * acc),end = '') 


# 先看一下模型的识别能力，可以看到没有经过训练的模型在测试集上的表现是很差的，大概只有x%左右的正确识别率
test(1)

### 训练！！！ 并在每个epoch之后测试 ###

# 根据epoch数正式训练并在每个epoch训练结束后测试
for epoch in range(1, n_epochs + 1):
    scheduler.step(test_acces[-1])
    train(epoch)
    test(epoch)
#输入最后保存的模型的准确率，也就是最高测试准确率
print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))
