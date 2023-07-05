import torch
import torchvision
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from vgg import VGG
from spp import spp_net, spp_layer
from misc import collate_fn
import json
import os
import time
import sys

from draw_cam2 import draw_CAM

import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
BATCH_SIZE = 16
EPOCH = 11
LR = 0.0001
print(DEVICE)

torch.cuda.empty_cache()

data_transform = {
    "train": transforms.Compose([transforms.Resize((256, 256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((256, 256)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 数据集处理及加载
# 获取图像数据集的路径
os.chdir(sys.path[0])
data_root = os.getcwd() 		# get data root path 返回上上层目录
image_path = data_root + "/sing_data/男x女/"  	
# image_path = data_root + "/sing_data/美声x民族/"  				 		


# 导入训练集并进行预处理
train_dataset = datasets.ImageFolder(root=image_path + "train",		
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,	# 导入的训练集
                                           batch_size=BATCH_SIZE, 	# 每批训练的样本数
                                           shuffle=True,	# 是否打乱训练集
                                           #collate_fn=collate_fn,
                                           num_workers=0)	# 使用线程数，在windows下设置为0

# 导入验证集并进行预处理
validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,	# 导入的验证集
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True,
                                              #collate_fn=collate_fn,
                                              num_workers=0)

# 获取分类的名称所对应的索引，即{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
sing_list = train_dataset.class_to_idx
print(train_dataset)
print(sing_list)
# 将 flower_list 中的 key 和 val 调换位置,使得预测之后返回的索引可以直接通过字典得到所属类别
cla_dict = dict((val, key) for key, val in sing_list.items())

# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
with open('sing_class_indices.json', 'w') as json_file:
    json_file.write(json_str)


#net = VGG(num_classes=2)                          # 实例化网络
net = spp_net(num_classes=2)
model_weight_path = os.path.abspath(os.path.join(os.getcwd(),'trained_models', "spp_美声民族.pkl"))
net.load_state_dict(torch.load(model_weight_path))
net.to(DEVICE)                                        # 分配网络到指定的设备（GPU/CPU）训练

loss_function = nn.CrossEntropyLoss()			 	  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=LR)	  # 优化器（训练参数，学习率）

prefix = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())).replace(":", "_").replace("-", "_")
save_path = prefix + '.pkl'
#save_path = './SPP_NET.pkl'
best_acc = 0.0

train_counter = []
train_losses = []

for epoch in range (EPOCH):
    ########################################## train ###############################################
    net.train()                 # 训练过程中开启 Dropout
    running_loss = 0.0			# 每个 epoch 都会对 running_loss 清零
    time_start = time.perf_counter()	# 对训练一个 epoch 计时

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        # print(data)
        
        images, labels = data   # 获取训练集的图像和标签
        optimizer.zero_grad()	# 清除历史梯度
        
        outputs = net(images.to(DEVICE))				 # 正向传播
        loss = loss_function(outputs, labels.to(DEVICE)) # 计算损失
        loss.backward()								     # 反向传播
        optimizer.step()								 # 优化器更新参数
        running_loss += loss.item()

        train_losses.append(loss.item())
        train_counter.append((step*BATCH_SIZE) + ((epoch-1)*len(train_loader.dataset)))

        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:.3f} [{}->{}] {:^3.0f}%".format(loss, a, b, int(rate * 100)), end="")
    print()
    print('%f s' % (time.perf_counter()-time_start))

    ########################################## validate ###############################################
    net.eval()    # 验证过程中关闭 Dropout
    acc = 0.0
    # with torch.no_grad():
    c = 0
    for val_data in validate_loader:
        
        val_images, val_labels = val_data

        outputs = net(val_images.to(DEVICE))   
        predict_y = torch.max(outputs, dim=1)[1] # 以output中值最大位置对应的索引（标签）作为预测输出
        acc += (predict_y == val_labels.to(DEVICE)).sum().item()   
        
        ''' TRY GRAD_CAM '''
        if epoch == 10:
            Save_Path = 'mei_min_cam'
            save_path2 = 'mei_min_heatmap'
            draw_CAM(net, val_images.to(DEVICE), Save_Path, save_path2, c, visual_heatmap=True)
        c += 1 
        


    val_accurate = acc/val_num

    
    # 保存准确率最高的那次网络参数
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
            (epoch + 1, running_loss / step, val_accurate))

    # 保存训练过程中的loss变化情况
    # fig = plt.figure()
    # plt.plot(train_counter, train_losses, color='blue')
    # plt.legend('Train Loss', loc='upper right')
    # plt.xlabel('number of training examples')
    # plt.ylabel('loss')
    # plt.show()
    # plt.savefig("train_loss.svg")
os.system('pause')
print('Finished Training')
print("The best accuracy is: ", best_acc)

# Class Activation Mapping (CAM)

