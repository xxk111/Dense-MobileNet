import os
import json
import time
from importlib_metadata import version
from matplotlib.colors import Normalize

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

#from model_v2 import MobileNetV2
from Dense_MobileNetv_1 import Dense_MobileNet_1
from Dense_MobileNetv_2 import Dense_MobileNet_2
from Dense_MobileNetv_3 import Dense_MobileNet_3

#预处理方法：  RandomResizedCrop(n) 随机裁剪n*n
#            RandomHorizontalFlip() 随即反转
#            ToTensor() 转化为Tensor
#            Normalize() 标准化
#            Resize() 改变大小

version = 1

def main():
    #查看是否有GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_path = "../train_00058.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(
        img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 实例化模型
    if version == 1:
        model = Dense_MobileNet_1(num_classes = 7)
    elif version == 2:
        model = Dense_MobileNet_2(num_classes = 7)
    elif version == 3:
        model = Dense_MobileNet_3(num_classes = 7)
    print('当前网络模型为 Dense-MobileNet-{}'.format(version))
    # load model weights
    model_weight_path = "result/dataSet_RAF2021-12-24_22-27-34/model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    start = time.time()
    output = model(img.to(device))
    end = time.time()
    with torch.no_grad():
        output = torch.squeeze((output)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        

    print("spend time:",end - start) 
    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
