import torch
import torch.nn as nn
import torch.optim as optim

def custom_resnet():
    # resnet101
    resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

    resnet_model._modules.pop('fc') #1000 fc

    resnet_model.fc1 = nn.Linear(2048, 15)
    # resnet_model.fc1 = nn.Linear(2048, 15)
    resnet_model.fc2 = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(15, 1)
    )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 2048*7*7

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    # add new_forward function to the resnet instance as a class method
    bound_method = forward.__get__(resnet_model, resnet_model.__class__)
    setattr(resnet_model, 'forward', bound_method)

    return resnet_model


def custom_resnet_optimizer(resnet_model):
    optimizer = optim.Adam(resnet_model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001)
    return optimizer