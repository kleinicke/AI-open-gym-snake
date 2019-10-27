"""
    AI Project by Florian Kleinicke
    Q-learning for Snake in an OpenAI/PLE environment

    The models used in the project.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


"""
    The base model I started with and did the most experients with
"""
class DQN(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        if self.activateAdditional:
            self.affine1 = nn.Linear(1072, 512)
        else:
            self.affine1 = nn.Linear(1024, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h=h.view(h.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)
            h = torch.cat((h, y), 1)

        h = F.relu(self.affine1(h))
        h = self.affine2(h)
        return h

"""
    This is the base model with on less convolutional layer
"""
class DQN_smaller(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_smaller, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        if self.activateAdditional:
            self.affine1 = nn.Linear(2352, 512)
        else:
            self.affine1 = nn.Linear(2304, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h=h.view(h.size(0), -1)

        if self.activateAdditional:
            y=y.view(h.size(0), -1)
            h = torch.cat((h, y), 1)


        h = F.relu(self.affine1(h))
        h = self.affine2(h)
        return h

"""
    this only got two convolutional layer and one fc layer
"""
class DQN_verysmall(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_verysmall, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0)
        if self.activateAdditional:
            self.affine2 = nn.Linear(1200, self.n_action)
        else:
            self.affine2 = nn.Linear(1152, self.n_action)


    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h=h.view(h.size(0), -1)

        if self.activateAdditional:
            y=y.view(h.size(0), -1)
            h = torch.cat((h, y), 1)

        h = self.affine2(h)
        return h

"""
    here the channels of the conv. layer were only 16. Also has an additional fc layer for the input from the additional features.
"""
class DQN_tiny(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_tiny, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=0)
        if self.activateAdditional:
            self.preprocess = nn.Linear(12*4 , 144)
            self.affine2 = nn.Linear(208 , self.n_action)
        else:
            self.affine2 = nn.Linear(64, self.n_action)


    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h=h.view(h.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)
            y=F.relu(self.preprocess(y))
            h = torch.cat((h, y), 1)

        h = self.affine2(h)
        return h

"""
    this network also had been used in the first try
"""
class DQN_other(nn.Module):
    def __init__(self,n_action,activateAdditional):
        super(DQN_other, self).__init__()
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 16, kernel_size=5,stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        if self.activateAdditional:
            self.fc1   = nn.Linear(1200,256)
        else:
            self.fc1   = nn.Linear(1152,256)
        self.fc2   = nn.Linear(256, 4)

    def forward(self, x,y):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x=x.view(x.size(0), -1)
        if self.activateAdditional:
            y=y.view(x.size(0), -1)
            x = torch.cat((x, y), 1)

        x=F.relu(self.fc1(x))

        return self.fc2(x.view(x.size(0), -1))

"""
    a bigger network 3 conv and 3 fc layer
"""
class DQN_big(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_big, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=4, padding=0)

        if self.activateAdditional:
            self.affine1 = nn.Linear(8240, 2048)
        else:
            self.affine1 = nn.Linear(8192, 2048)
        self.affine12 = nn.Linear(2048, 512)

        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h=h.view(h.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)
            h = torch.cat((h, y), 1)

        h = F.relu(self.affine1(h))
        h = F.relu(self.affine12(h))
        h = self.affine2(h)
        return h

"""
    this network is smaller than DQN_big. 3 conv and 2 fc layer
"""
class DQN_big2(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_big2, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=7, stride=4, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=4, padding=0)

        if self.activateAdditional:
            self.affine1 = nn.Linear(336, 64)
        else:
            self.affine1 = nn.Linear(288, 64)

        self.affine2 = nn.Linear(64, self.n_action)

    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h=h.view(h.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)
            h = torch.cat((h, y), 1)

        h = F.relu(self.affine1(h))
        h = self.affine2(h)
        return h


"""
    test to compare 0 and 1 padding. test2 has 1 padding in first conv layer.
"""
class DQN_test1(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_test1, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=7, stride=4, padding=0)

        convelems=6272
        if self.activateAdditional:
            self.affine1 = nn.Linear(convelems+12, 256)
        else:
            self.affine1 = nn.Linear(convelems, 256)

        self.affine2 = nn.Linear(256, self.n_action)

    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))

        h=h.view(h.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)
            h = torch.cat((h, y), 1)

        h = F.relu(self.affine1(h))
        h = self.affine2(h)
        return h

"""
    same as test1, except for the 1 padding in first conv layer
"""
class DQN_test2(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_test2, self).__init__()
        self.n_action = 4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=7, stride=4, padding=0)

        convelems=7200
        if self.activateAdditional:
            self.affine1 = nn.Linear(convelems+12, 256)
        else:
            self.affine1 = nn.Linear(convelems, 256)

        self.affine2 = nn.Linear(256, self.n_action)


    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))

        h=h.view(h.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)

            h = torch.cat((h, y), 1)

        h = F.relu(self.affine1(h))
        h = self.affine2(h)
        return h

"""
    sadly didn't learn anything
"""
class DQN_insp(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_insp, self).__init__()

        self.n_action = 4#n_action#4
        self.activateAdditional=activateAdditional

        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        convsize=512
        if self.activateAdditional:
            self.affine1 = nn.Linear(convsize+48, self.n_action)
        else:
            self.affine1 = nn.Linear(convsize, self.n_action)


    def forward(self, x,y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h=h.view(h.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)

            h = torch.cat((h, y), 1)

        h = F.relu(self.affine1(h))

        return h

"""
    only fully connected layer
"""
class DQN_fully(nn.Module):
    def __init__(self, n_action,activateAdditional):
        super(DQN_fully, self).__init__()
        self.n_action = 4#n_action#4
        self.activateAdditional=activateAdditional

        convsize=16384
        if self.activateAdditional:
            self.affine1 = nn.Linear(convsize+48, 256)
        else:
            self.affine1 = nn.Linear(convsize, 256)

        self.affine2 = nn.Linear(256, self.n_action)

    def forward(self, x,y):

        h=x.view(x.size(0), -1)
        if self.activateAdditional:
            y=y.view(h.size(0), -1)

            h = torch.cat((h, y), 1)

        h = F.relu(self.affine1(h))
        h = self.affine2(h)
        return h
