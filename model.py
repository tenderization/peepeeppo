import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, channels, h, w):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.norm1 = torch.nn.InstanceNorm2d(channels)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.norm2 = torch.nn.InstanceNorm2d(channels)
        self.act2 = torch.nn.ReLU()

    def forward(self, x):
        return x + self.act2(self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x))))))
        #return x + self.act2(self.conv2(self.act1(self.conv1(x))))

class ConvNet(torch.nn.Module):
    def __init__(self, h, w, channels=32, layers=2, d=128, p=9):
        super().__init__()
        self.h = h
        self.w = w
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.InstanceNorm2d(channels)
        self.act1 = torch.nn.ReLU()
        self.convs = torch.nn.Sequential(*[ConvBlock(channels, h, w) for _ in range(layers)])
        self.mlp = torch.nn.Linear(h*w*channels, d)
        self.act_final = torch.nn.ReLU()
        self.policy_head = torch.nn.Linear(d, p)
        self.value_head = torch.nn.Linear(d, 1)
        
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x.reshape(-1, 1, self.h, self.w))))
        #x = self.act1(self.conv1(x.reshape(-1, 1, self.h, self.w)))
        x = self.convs(x)
        x = self.mlp(x.reshape(-1, self.h*self.w*self.channels))
        x = self.act_final(x)
        return self.policy_head(x), self.value_head(x)
