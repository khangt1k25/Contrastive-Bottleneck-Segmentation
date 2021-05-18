import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out



def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
}


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat



class MaskGenerator(nn.Sequential):
    def __init__(self, dataset, out_channels=1):
        super(MaskGenerator, self).__init__()

        # For MNIST
        if dataset == "mnist":
            input_channel = 1
            channel_init = 16
            steps = 2
		# For CIFAR
        else:
            input_channel = 3
            channel_init = 32
            steps = 3

        # channel_current is 'in_channel'
        # channel_next is 'out_channel'
        channel_current = input_channel
        channel_next = channel_init

        for step in range(steps):
            self.add_module("convblock_down_{}".format(2 * step),
                            Conv2dBlock(channel_current, channel_next))
            self.add_module("convblock_down_{}".format(2 * step + 1),
                            Conv2dBlock(channel_next, channel_next))
            self.add_module("downsample_{}".format(step), DownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self.add_module("convblock_middle", Conv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module("upsample_{}".format(step), UpSampleBlock())
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))

            # Add another convblock_up. However, if the layer is the last layer, do not use ReLU!
            if step == steps - 1:
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))

            channel_current = channel_next
            channel_next = channel_next // 2

            if step == steps - 2:
                if out_channels is None:
                    channel_next = input_channel
                else:
                    channel_next = out_channels


    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / 2 + 0.5
        x = torch.round(x)
        return x