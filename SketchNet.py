import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def fc_block(size_in, size_out):
    return nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )

class SketchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # (N, 6, 256, 256)
            conv_block(6, 64),                     # (N, 64, 256, 256)
            conv_block(64, 64),                    # (N, 64, 256, 256)
            nn.MaxPool2d(kernel_size=2, stride=2), # (N, 64, 128, 128)
            conv_block(64, 128),                   # (N, 128, 128, 128)
            conv_block(128, 128),                  # (N, 128, 128, 128)
            nn.MaxPool2d(kernel_size=2, stride=2), # (N, 128, 64, 64)
            conv_block(128, 256),                  # (N, 256, 64, 64)
            conv_block(256, 256),                  # (N, 256, 64, 64)
            conv_block(256, 256),                  # (N, 256, 64, 64)
            nn.MaxPool2d(kernel_size=2, stride=2), # (N, 256, 32, 32)
            conv_block(256, 512),                  # (N, 512, 32, 32)
            conv_block(512, 512),                  # (N, 512, 32, 32)
            conv_block(512, 512),                  # (N, 512, 32, 32)
            nn.MaxPool2d(kernel_size=2, stride=2), # (N, 512, 16, 16)
            conv_block(512, 512),                  # (N, 512, 16, 16)
            conv_block(512, 512),                  # (N, 512, 16, 16)
            conv_block(512, 512),                  # (N, 512, 16, 16)
            nn.MaxPool2d(kernel_size=2, stride=2)  # (N, 512, 8, 8)
        )
        self.fc1 = fc_block(512*8*8, 4096)
        self.fc2 = fc_block(4096, 4096)
        
        self.fc3 = fc_block(4099, 1024)
        self.fc4 = fc_block(1024, 1024)
        self.output_fc = nn.Sequential(
            nn.Linear(1024, 3),
            nn.Tanh()
        )
    
    def forward(self, image, brush_pos):
        # image: (N, 6, 256, 256)
        # brush_pos: (N, 3)
        tmp = self.conv(image)            # (N, 512, 8, 8)
        tmp = tmp.view(tmp.size(0), -1) # (N, 512*8*8)
        tmp = self.fc1(tmp)               # (N, 4096)
        image_features = self.fc2(tmp)    # (N, 4096)
        
        x = torch.cat([image_features, brush_pos], dim=1) # (N, 4099)
        x = self.fc3(x)       # (N, 1024)
        x = self.fc4(x)       # (N, 1024)
        x = self.output_fc(x) # (N, 3)
        
        x = x * torch.Tensor([1, 1, 0.2]).to(x.device)
        return x
