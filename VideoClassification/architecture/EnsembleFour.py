import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchinfo import summary

class DeepSearchConv(torch.nn.Module):
    def __init__(self, inputs, outputs, kernel_size):
        super(DeepSearchConv, self).__init__()
        
        self.conv = nn.Conv2d(inputs, outputs, kernel_size = kernel_size)
        self.batch_norm = nn.BatchNorm2d(outputs)
        self.dropout = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return x
    
class DeepSearchLinear(torch.nn.Module):
    def __init__(self, inputs, outputs):
        super(DeepSearchLinear, self).__init__()
        
        self.linear = nn.Linear(inputs, outputs)
        nn.init.xavier_uniform_(self.linear.weight)
        self.batch_norm = nn.BatchNorm1d(outputs)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return x
    
class DeepSearch(torch.nn.Module):
    def __init__(self, hidden, kernel_size, window, hidden_linear, labels=2):
        super(DeepSearch, self).__init__()
        
        self.conv = [DeepSearchConv(hidden[i], hidden[i+1],  
                                    kernel_size) for i in range(len(hidden)-1)]
        self.conv_combined = nn.Sequential(*self.conv)
        
        self.avgpool = nn.AvgPool2d(window)
        self.flattened = hidden_linear[0]
        self.linear= [DeepSearchLinear(hidden_linear[i], hidden_linear[i+1]) for i in range(len(hidden_linear)-1)]
        self.linear_combined = nn.Sequential(*self.linear)
        resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        alexnet = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.DEFAULT)
        convexnet = torchvision.models.convnext_base(weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT)
        googlenet = torchvision.models.googlenet(weights =torchvision.models.GoogLeNet_Weights.DEFAULT)

        # Telling the model to keep pre-trained weights
        for param in resnet.parameters():    
            param.requires_grad = False
        for param in alexnet.parameters():
            param.requires_grad = False
        for param in convexnet.parameters():
            param.requires_grad = False
        for param in googlenet.parameters():
            param.requires_grad = False
        self.model_1 = resnet
        self.model_2 = alexnet
        self.model_3 = convexnet
        self.model_4 = googlenet
        
        self.output = nn.Linear(hidden_linear[-1], labels)
    
    def forward(self, x):
        x_1 = self.model_1(x)
        x_2 = self.model_2(x)
        x_3 = self.model_3(x)
        x_4 = self.model_4(x)
        x = x_1+ x_2 +x_3 +x_4
        x = self.conv_combined(x)
        x = self.avgpool(x)
        x = x.view(-1, self.flattened)
        x = self.linear_combined(x)
        x = self.output(x)
        return x
