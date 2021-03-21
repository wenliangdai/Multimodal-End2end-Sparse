from torch import nn

class VggBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(VggBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
