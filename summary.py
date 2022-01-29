#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from torchsummary import summary
from torchstat import stat
from Dense_MobileNetv_1 import Dense_MobileNet_1
from Dense_MobileNetv_2 import Dense_MobileNet_2
from Dense_MobileNetv_3 import Dense_MobileNet_3

version = '1'

if __name__ == "__main__":
    if version == '1':
        model = Dense_MobileNet_1(num_classes = 7)
    elif version == '2':
        model = Dense_MobileNet_2(num_classes = 7)
    elif version == '3':
        model = Dense_MobileNet_3(num_classes = 7)
    stat(model, (3, 224, 224))
    # summary(model, (3, 224, 224))
