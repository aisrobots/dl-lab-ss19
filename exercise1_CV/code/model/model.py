import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls


class ResNetConv(ResNet):    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        intermediate = []
        x = self.layer1(x); intermediate.append(x)
        x = self.layer2(x); intermediate.append(x)
        x = self.layer3(x); intermediate.append(x)
        
        return x, intermediate


class ResNetModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        
        # other network modules
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 34)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, inputs, filename):
        x, _ = self.res_conv(inputs)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + 0.5
        return x
