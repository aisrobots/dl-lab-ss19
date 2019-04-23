import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        model = models.resnet18(pretrained=pretrained)
        self.res_conv = nn.Sequential(*list(model.children())[:-3])  # discards last three layers.

        # other network modules
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 34)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs, filename):
        x = self.res_conv(inputs)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + 0.5
        return x
