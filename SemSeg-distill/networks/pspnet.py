import torch
import torch.nn as nn
from torch.nn import functional as F
affine_par = True
import functools
from inplace_abn import InPlaceABN, InPlaceABNSync
import math

from networks.resnet import resnet18, resnet101

#BatchNorm2d = functools.partial(InPlaceABN, activation='identity')
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')



class TransferConv(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature), nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, student):
        student = self.Connectors(student)
        return student

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(BasicBlock, self).__init__()
        dilation = dilation*multi_grid
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) 
                  for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, pretrained):
        self.inplanes = 128
        super(ResNet, self).__init__()
        
        if layers == [2, 2, 2, 2]:
            resnet = resnet18(pretrained=pretrained)
        elif layers == [3, 4, 23, 3]:
            resnet = resnet101(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, 
                                    resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        if layers == [3, 4, 23, 3]:
            self.pspmodule = PSPModule(2048, 512)
            self.head = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            self.dsn = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                InPlaceABNSync(512),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
        elif layers == [2, 2, 2, 2]:
            self.pspmodule = PSPModule(512, 128)
            self.head = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            self.dsn = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                InPlaceABNSync(128),
                nn.Dropout2d(0.1),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
        else:
            raise ValueError('layers should be [3, 4, 23, 3] or [2, 2, 2, 2]')

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, feats=None, feat_layer='last'):
        if feats is not None:
            x_dsn = None
            if feat_layer == 'last':
                x_feat = None
                x_feat_after_psp = feats
            else:
                x_feat = feats
                x_feat_after_psp = self.pspmodule(x_feat)
        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x_dsn = self.dsn(x)
            x_feat = self.layer4(x)
            #x = self.head(x4)
            x_feat_after_psp = self.pspmodule(x_feat)
            
        x = self.head(x_feat_after_psp)
        return [x, x_dsn, x_feat, x_feat_after_psp]

def Res_pspnet(block = Bottleneck, layers = [3, 4, 23, 3], num_classes=21, pretrained=True):
    '''
    ResNet(Bottleneck, [3, 4, 23, 3], num_classes) -> 101
    ResNet(BasicBlock, [2, 2, 2, 2], num_classes)  -> 18
    '''
    model = ResNet(block, layers, num_classes, pretrained)
    return model
