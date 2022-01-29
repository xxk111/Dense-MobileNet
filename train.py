import os
import json
import time
import getopt
import sys
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from Dense_MobileNetv_1 import Dense_MobileNet_1
from Dense_MobileNetv_2 import Dense_MobileNet_2
from Dense_MobileNetv_3 import Dense_MobileNet_3

#选择数据集 RAF test
DATA_SET = 'test'
#选择Dense-MobileNet类型 1 2 3
version = 1
#设定BatchSize和epochs
batch_size = 16
epochs = 60


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("using {} device.".format(device))
    #新建输出文件夹
    now = time.strftime("{}_%Y-%m-%d_%H-%M-%S".format(DATA_SET), time.localtime(time.time()))
    root = os.path.abspath('.')
    result_dir = os.path.join(root, 'result', now)
    os.mkdir(result_dir)
    #定义预处理
    data_transform = {
        "train":
        transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("---------------读入数据集------------------\n")
    data_root = os.path.abspath(os.path.join(os.getcwd(),".."))  # 去到数据集根目录
    image_path = os.path.join(data_root, "data_set",DATA_SET)  # 数据集路径
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #训练集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),transform=data_transform["train"])
    train_num = len(train_dataset)
    #标签
    face_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in face_list.items())
    # 写入json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)

    print("训练集数量：{} , 测试集数量：{}\n".format(train_num, val_num))

    # 实例化模型
    if version == 1:
        net = Dense_MobileNet_1(num_classes = 7)
    elif version == 2:
        net = Dense_MobileNet_2(num_classes = 7)
    elif version == 3:
        net = Dense_MobileNet_3(num_classes = 7)
    print('当前网络模型为 Dense-MobileNet-{}'.format(version))
    net.to(device)

    #定义损失函数
    loss_function = nn.CrossEntropyLoss()

    #优化器
    optimizer = optim.Adam(net.parameters(),lr=1e-3) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    #保存参数
    best_acc = 0.0
    train_loss_list = []
    val_accuracy_list = []

    train_steps = len(train_loader)
    print("--------------开始训练---------------\n")
    for epoch in range(epochs):
        #训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        #测试
        scheduler.step()
        net.eval()
        acc = 0.0  # accumulate accurate: number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        val_accurate = acc / val_num
        train_loss = running_loss / train_steps
        train_loss_list.append(train_loss)
        val_accuracy_list.append(val_accurate)
        print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f' %(epoch + 1, train_loss, val_accurate))
        SAVE_PATH = os.path.join(result_dir, 'epoch{} model.pth'.format(epoch + 1))
        torch.save(net.state_dict(), SAVE_PATH)    
        if val_accurate > best_acc:
            best_acc = val_accurate

    #保存模型
    
    #打印参数
    print('----------------训练完成-------------------\n')
    print('最高精确率为：{:.3f}'.format(best_acc))
    print('准确率list：',val_accuracy_list)
    print('损失list：',train_loss_list)



if __name__ == '__main__':
    main()

