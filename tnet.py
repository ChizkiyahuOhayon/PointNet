# -- coding: utf-8 -
import torch
import torch.nn as nn
import torch.functional as F

# (n, 3) -> (n, 64) -> (n, 128) -> (n, 1024) --max_pooling--> (n)
# --> (n) --> (9) --> (3, 3) + identity matrix --> (3, 3)


class TNet(nn.Module):
    def __init__(self, init_dim=3, max_dim=1024, num_point=2000):
        super(TNet, self).__init__()

        self.init_dim = init_dim
        self.max_dim = max_dim
        # 1. increase dimension
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, max_dim, kernel_size=1)

        # 2. got global feature
        self.max_pooling = nn.MaxPool1d(kernel_size=num_point)

        # 3. decrease the dimension
        self.linear1 = nn.Linear(max_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, init_dim ** 2)

        # 4. add an identity matrix to it

        self.bn1 = nn.BatchNorm1d(64) # 64 mean values, 64 covariance
        self.bn2 =  nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batchsize, 2000, 3)
        print(x.shape)
        batchsize = x.shape[0]

        # increase dimensions through conv1d
        x = self.bn1(self.relu(self.conv1(x)))
        print(x.shape)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        # (batchsize, 2000, 1024)

        # max pooling
        x = self.max_pooling(x)  # (batchsize, 2000, 1)
        x = x.view(batchsize, -1)  # (batchsize, 2000)

        # decrease the dimension
        x = self.bn4(self.relu(self.linear1(x)))
        x = self.bn5(self.relu(self.linear2(x)))
        x = self.linear3(x)   # (batchsize, 9)

        # add an identity matrix to it
        """
        1 0 0
        0 1 0
        0 0 1
        """
        iden_3x3 = torch.eye(self.init_dim, requires_grad=True).repeat(batchsize, 1, 1)
        if x.is_cuda:
            iden_3x3 = iden_3x3.cuda()

        x = x.view(-1, self.init_dim, self.init_dim) + iden_3x3

        return x

if __name__ == "__main__":
    input = torch.ones(16, 3, 2000)
    print(input.shape)
    model = TNet()
    output = model(input)
    print(output.shape)