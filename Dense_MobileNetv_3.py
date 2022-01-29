import torch
import torch.nn as nn

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )
    
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )

class DenseBlock(nn.Module):
    def __init__(self, in_channel):
        """
        每次卷积输入输出的模块维度是相同的, 最后拼接在一起
        :param in_channel:输入维度, 输出维度相同
        """
        super(DenseBlock, self).__init__()
        self.d1 = conv_dw(in_channel, in_channel,1)
        self.d2 = conv_dw(2 * in_channel, in_channel,1)
        self.d3 = conv_dw(3 * in_channel, in_channel,1)
        self.d4 = conv_dw(4 * in_channel, in_channel,1)
        self.Transition = self.Transition_Layer(in_channel*5 ,in_channel)

    def Transition_Layer(self, in_, out):
        """
        控制升维和降维
        :param in_:
        :param out:
        :return:
        """
        transition = nn.Sequential(
            nn.BatchNorm2d(in_),
            nn.ReLU(),
            nn.Conv2d(in_, out, 1),
        )
        return transition

    def forward(self, x):
        x1 = self.d1(x)
        x_cat1 = torch.cat((x, x1), dim=1)
        x2 = self.d2(x_cat1)
        x_cat2 = torch.cat((x2, x_cat1), dim=1)
        x3 = self.d3(x_cat2)
        x_cat3 = torch.cat((x3, x_cat2), dim=1)
        x4 = self.d4(x_cat3)
        x = torch.cat((x4,x_cat3), dim=1)
        return x

class Dense_MobileNet_3(nn.Module):
    def __init__(self,num_classes=7):
        super(Dense_MobileNet_3, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 2), 
            conv_dw(32, 32, 1), 
            conv_dw(32, 32, 2),
            DenseBlock(32),
            conv_dw(160, 64, 2),
            DenseBlock(64),
            conv_dw(320, 128, 2),
            DenseBlock(128),
            conv_dw(640, 1024, 2),
            conv_dw(1024, 1024, 1),
        )    
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = Dense_MobileNet()
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)