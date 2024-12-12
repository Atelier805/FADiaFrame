import torch
import torch.nn as nn
from torchvision import models

class Calibration_Model(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(Calibration_Model, self).__init__()
        self.drop_rate = dropout
        self.fc1 = nn.Linear(1, 2048)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.fc2 = nn.Linear(2048, 2048)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(2048)
        self.dropout_bn = nn.Dropout(self.drop_rate)
        self.layernorm1 = nn.LayerNorm(2048)
        self.layernorm2 = nn.LayerNorm(2048)
        self.dropout_ln = nn.Dropout(self.drop_rate)
        self.fc3 = nn.Linear(2048, 1)
        self.dropout3 = nn.Dropout(self.drop_rate)
        self.fc4 = nn.Linear(2048, 32) 
        self.dropout4 = nn.Dropout(self.drop_rate)


    def forward(self, x1, x2):
        
        x1 = self.fc1(x1)
        x1 = self.dropout1(x1)

        x2 = self.fc2(x2)
        x2 = self.dropout2(x2)

        x = self.layernorm1(x1) + self.layernorm2(x2)
        x = self.dropout_ln(x)
        
        x = self.bn(x)
        x = self.dropout_bn(x)
        x = self.gelu(x)

        x_2048 = x
        x_32 = self.fc4(x_2048)
        x_32 = self.dropout4(x_32)

        x = self.fc3(x)
        x = self.dropout3(x)

        return x
