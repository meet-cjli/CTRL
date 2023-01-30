import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
from .squeezenet import cifar_sqnxt_23_1x
import math
import time

def conv_bn(inp, oup, stride, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2

        if self.benchmodel == 1:
            # Figure 3(c)
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            # Figure 3(d)
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1== self.benchmodel:
            # Figure 3(c)
            x1 = x[:, :(x.shape[1] // 2), :, :]  # first branch
            x2 = x[:, (x.shape[1] // 2):, :, :]  # second branch
            out = self._concat(x1, self.banch2(x2))  # concat
        elif 2 == self.benchmodel:
            # Figure 3(d)
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=1, padding=1)  # 32x32
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 8)))

        # building classifier
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))
        # self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)  # 32x32x24
        # x = self.maxpool(x)
        x = self.features(x)  # 16x16x116=> 8x8x232 => 4x4x464
        x = self.conv_last(x)  # 4x4x1024
        x = self.globalpool(x)  # 1x1x1024
        x = x.view(-1, self.stage_out_channels[-1])
        #x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def speed(model, name, inputX, inputY):
    t0 = time.time()
    input = torch.rand(1, 3, inputX, inputY).cuda()
    input = Variable(input, volatile=True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    print('=> {} cost: {}'.format(name, t2 - t1))


def cifar_shufflenet_v2_0_5x_32(num_classes=10):
    model = ShuffleNetV2(n_class=num_classes, input_size=32, width_mult=0.5)
    return model


def cifar_shufflenet_v2_1x_32(num_classes=10):
    model = ShuffleNetV2(n_class=num_classes, input_size=32, width_mult=1.)
    return model


def cifar_shufflenet_v2_1_5x_32(num_classes=10):
    model = ShuffleNetV2(n_class=num_classes, input_size=32, width_mult=1.5)
    return model


def cifar_shufflenet_v2_2x_32(num_classes=10):
    model = ShuffleNetV2(n_class=num_classes, input_size=32, width_mult=2.)
    return model


if __name__ == '__main__':
    """Testing
    """
    # model = cifar_shufflenet_v2_0_5x_32(num_classes=10).cuda()
    # # print("=> ShuffleNetV2 0.5x 32:\n {}".format(model))
    # speed(model, 'ShuffleNetV2 0.5x 32', 32, 32) # for 32x32
    # print("=> ShuffleNetV2 0.5x 32 param : {}".format(count_parameters(model)))

    #model = cifar_shufflenet_v2_1x_32(num_classes=10).cuda()
    model = cifar_sqnxt_23_1x().cuda()
    #model = mobilenet_v2_0_4x_32().cuda()
    print("=> ShuffleNetV2 1x 224:\n {}".format(model))
    speed(model, 'ShuffleNetV2 1x 32', 32, 32)  # for 32x32
    print("=> ShuffleNetV2 1x 32 param : {}".format(count_parameters(model)))

    # model = cifar_shufflenet_v2_1_5x_32(num_classes=10).cuda()
    # # print("=> ShuffleNetV2 1.5x 32:\n {}".format(model))
    # speed(model, 'ShuffleNetV2 1.5x 32', 32, 32) # for 32x32
    # print("=> ShuffleNetV2 1.5x 32 param : {}".format(count_parameters(model)))

    # model = cifar_shufflenet_v2_2x_32(num_classes=10).cuda()
    # # print("=> ShuffleNetV2 2x 32:\n {}".format(model))
    # speed(model, 'ShuffleNetV2 2x 32', 32, 32) # for 32x32
    # print("=> ShuffleNetV2 2x 32 param : {}".format(count_parameters(model)))