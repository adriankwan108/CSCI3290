import torch
import torch.nn as nn
import torch.nn.functional as F

"""
#upsample
input, target = data[0].to(device), data[1].to(device)
input = F.upsample(input, scale_factor=target.shape[1], mode='bicubic')
"""

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        ######################
        # write your code here
        #patch extraction  #output 64 features 
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, padding = 4)
        #non-linear mapping  #64 features with 32 filters of 1*1
        self.conv2 = nn.Conv2d(64,32, kernel_size = 1, padding=0)
        #reconstruction
        self.conv3 = nn.Conv2d(32,3, kernel_size = 5, padding= 2)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        ######################
        # write your code here
        x = F.interpolate(x, scale_factor=3, mode='bicubic', align_corners=True )
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x
