import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.jit import script


class WSConv2d(nn.Conv2d):
    def __init___(self, in_channels, out_channels, kernel_size, stride=1,
                  padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        # std = torch.sqrt(torch.var(weight.view(weight.size(0),-1),dim=1)+1e-12).view(-1,1,1,1)+1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv_ws(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return WSConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                    groups=groups, bias=bias)


'''
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x*torch.tanh(F.softplus(x))
'''


@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))


@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)


# Cell
def mish(x): return MishJitAutoFn.apply(x)


class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


######################################################################################################################
######################################################################################################################

# pre-activation based upsampling conv block
class upConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, norm, act, num_groups):
        super(upConvLayer, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                         bias=False)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = act
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)  # pre-activation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv(x)
        return x


# pre-activation based conv block
class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1,
                 padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN':
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride,
                           padding=padding, dilation=dilation, groups=1, bias=bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        out = self.module(x)
        return out


# Deep Feature Fxtractor
class deepFeatureExtractor_ResNext101(nn.Module):
    def __init__(self, args, lv6=False):
        super(deepFeatureExtractor_ResNext101, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
        self.fixList = ['layer1.0', 'layer1.1', '.bn']
        self.lv6 = lv6

        if lv6 is True:
            self.layerList = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.dimList = [64, 256, 512, 1024, 2048]
        else:
            del self.encoder.layer4
            del self.encoder.fc
            self.layerList = ['relu', 'layer1', 'layer2', 'layer3']
            self.dimList = [64, 256, 512, 1024]

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            # feature = v(features[-1])
            # features.append(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable


# ASPP Module
class Dilated_bottleNeck(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck, self).__init__()
        conv = conv_ws
        # in feat = 1024 in ResNext101 and ResNet101
        self.reduction1 = conv(in_feat, in_feat // 2, kernel_size=1, stride=1, bias=False, padding=0)
        self.aspp_d3 = nn.Sequential(
            myConv(in_feat // 2, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 2) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=3, dilation=3, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d6 = nn.Sequential(
            myConv(in_feat // 2 + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat // 2 + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=6, dilation=6, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d12 = nn.Sequential(
            myConv(in_feat, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=12, dilation=12, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.aspp_d18 = nn.Sequential(
            myConv(in_feat + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=18, dilation=18, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.reduction2 = myConv(((in_feat // 4) * 4) + (in_feat // 2), in_feat // 2, kSize=3, stride=1, padding=1,
                                 bias=False, norm=norm, act=act, num_groups=((in_feat // 4) * 4 + (in_feat // 2)) // 16)

    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12], dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x, d3, d6, d12, d18], dim=1))
        return out  # 512 x H/16 x W/16


class Dilated_bottleNeck2(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck2, self).__init__()
        conv = conv_ws
        # in feat = 1024 in ResNext101 and ResNet101
        # self.reduction1 = conv(in_feat, in_feat//2, kernel_size=1, stride = 1, bias=False, padding=0)
        self.reduction1 = conv(in_feat, in_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.aspp_d3 = nn.Sequential(
            myConv(in_feat // 2, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 2) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=3, dilation=3, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d6 = nn.Sequential(
            myConv(in_feat // 2 + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat // 2 + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=6, dilation=6, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d12 = nn.Sequential(
            myConv(in_feat, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=12, dilation=12, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.aspp_d18 = nn.Sequential(
            myConv(in_feat + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=18, dilation=18, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.aspp_d24 = nn.Sequential(
            myConv(in_feat + in_feat // 2, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat + in_feat // 2) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=24, dilation=24, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.reduction2 = myConv(((in_feat // 4) * 5) + (in_feat // 2), in_feat // 2, kSize=3, stride=1, padding=1,
                                 bias=False, norm=norm, act=act, num_groups=((in_feat // 4) * 5 + (in_feat // 2)) // 16)

    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12], dim=1)
        d18 = self.aspp_d18(cat3)
        cat4 = torch.cat([cat3, d18], dim=1)
        d24 = self.aspp_d24(cat4)
        out = self.reduction2(torch.cat([x, d3, d6, d12, d18, d24], dim=1))
        return out  # 512 x H/16 x W/16


class Dilated_bottleNeck_lv6(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck_lv6, self).__init__()
        conv = conv_ws
        in_feat = in_feat // 2
        self.reduction1 = myConv(in_feat * 2, in_feat // 2, kSize=3, stride=1, padding=1, bias=False, norm=norm,
                                 act=act, num_groups=(in_feat) // 16)
        self.aspp_d3 = nn.Sequential(
            myConv(in_feat // 2, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 2) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=3, dilation=3, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d6 = nn.Sequential(
            myConv(in_feat // 2 + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat // 2 + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=6, dilation=6, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d12 = nn.Sequential(
            myConv(in_feat, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=12, dilation=12, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.aspp_d18 = nn.Sequential(
            myConv(in_feat + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=18, dilation=18, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.reduction2 = myConv(((in_feat // 4) * 4) + (in_feat // 2), in_feat, kSize=3, stride=1, padding=1,
                                 bias=False, norm=norm, act=act, num_groups=((in_feat // 4) * 4 + (in_feat // 2)) // 16)

    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12], dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x, d3, d6, d12, d18], dim=1))
        return out  # 512 x H/16 x W/16


# Laplacian Decoder Network
class Lap_decoder_lv5(nn.Module):
    def __init__(self, args, dimList):
        super(Lap_decoder_lv5, self).__init__()
        norm = args.norm
        conv = conv_ws
        if norm == 'GN':
            if args.rank == 0:
                print("==> Norm: GN")
        else:
            if args.rank == 0:
                print("==> Norm: BN")

        if args.act == 'ELU':
            act = 'ELU'
        elif args.act == 'Mish':
            act = 'Mish'
        else:
            act = 'ReLU'
        kSize = 3
        self.max_depth = args.max_depth
        self.ASPP = Dilated_bottleNeck(norm, act, dimList[3])
        self.dimList = dimList
        ############################################     Pyramid Level 5     ###################################################
        # decoder1 out : 1 x H/16 x W/16 (Level 5)
        self.decoder1 = nn.Sequential(
            myConv(dimList[3] // 2, dimList[3] // 4, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 2) // 16),
            myConv(dimList[3] // 4, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 4) // 16),
            myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16),
            myConv(dimList[3] // 16, dimList[3] // 32, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 16) // 16),
            myConv(dimList[3] // 32, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 32) // 16)
            )
        ########################################################################################################################

        ############################################     Pyramid Level 4     ###################################################
        # decoder2 out : 1 x H/8 x W/8 (Level 4)
        # decoder2_up : (H/16,W/16)->(H/8,W/8)
        self.decoder2_up1 = upConvLayer(dimList[3] // 2, dimList[3] // 4, 2, norm, act, (dimList[3] // 2) // 16)
        self.decoder2_reduc1 = myConv(dimList[3] // 4 + dimList[2], dimList[3] // 4 - 4, kSize=1, stride=1, padding=0,
                                      bias=False,
                                      norm=norm, act=act, num_groups=(dimList[3] // 4 + dimList[2]) // 16)
        self.decoder2_1 = myConv(dimList[3] // 4, dimList[3] // 4, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 4) // 16)

        self.decoder2_2 = myConv(dimList[3] // 4, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 4) // 16)
        self.decoder2_3 = myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)

        self.decoder2_4 = myConv(dimList[3] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        ########################################################################################################################

        ############################################     Pyramid Level 3     ###################################################
        # decoder2 out2 : 1 x H/4 x W/4 (Level 3)
        # decoder2_1_up2 : (H/8,W/8)->(H/4,W/4)
        self.decoder2_1_up2 = upConvLayer(dimList[3] // 4, dimList[3] // 8, 2, norm, act, (dimList[3] // 4) // 16)
        self.decoder2_1_reduc2 = myConv(dimList[3] // 8 + dimList[1], dimList[3] // 8 - 4, kSize=1, stride=1, padding=0,
                                        bias=False,
                                        norm=norm, act=act, num_groups=(dimList[3] // 8 + dimList[1]) // 16)
        self.decoder2_1_1 = myConv(dimList[3] // 8, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)

        self.decoder2_1_2 = myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)

        self.decoder2_1_3 = myConv(dimList[3] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        ########################################################################################################################

        ############################################     Pyramid Level 2     ###################################################
        # decoder2 out3 : 1 x H/2 x W/2 (Level 2)
        # decoder2_1_1_up3 : (H/4,W/4)->(H/2,W/2)
        self.decoder2_1_1_up3 = upConvLayer(dimList[3] // 8, dimList[3] // 16, 2, norm, act, (dimList[3] // 8) // 16)
        self.decoder2_1_1_reduc3 = myConv(dimList[3] // 16 + dimList[0], dimList[3] // 16 - 4, kSize=1, stride=1,
                                          padding=0, bias=False,
                                          norm=norm, act=act, num_groups=(dimList[3] // 16 + dimList[0]) // 16)
        self.decoder2_1_1_1 = myConv(dimList[3] // 16, dimList[3] // 16, kSize, stride=1, padding=kSize // 2,
                                     bias=False,
                                     norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        self.decoder2_1_1_2 = myConv(dimList[3] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                     norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        ########################################################################################################################

        ############################################     Pyramid Level 1     ###################################################
        # decoder5 out : 1 x H x W (Level 1)
        # decoder2_1_1_1_up4 : (H/2,W/2)->(H,W)
        self.decoder2_1_1_1_up4 = upConvLayer(dimList[3] // 16, dimList[3] // 16 - 4, 2, norm, act,
                                              (dimList[3] // 16) // 16)
        self.decoder2_1_1_1_1 = myConv(dimList[3] // 16, dimList[3] // 16, kSize, stride=1, padding=kSize // 2,
                                       bias=False,
                                       norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        self.decoder2_1_1_1_2 = myConv(dimList[3] // 16, dimList[3] // 32, kSize, stride=1, padding=kSize // 2,
                                       bias=False,
                                       norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        self.decoder2_1_1_1_3 = myConv(dimList[3] // 32, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                       norm=norm, act=act, num_groups=(dimList[3] // 32) // 16)
        ########################################################################################################################
        self.upscale = F.interpolate

    def forward(self, x, rgb):
        cat1, cat2, cat3, dense_feat = x[0], x[1], x[2], x[3]
        rgb_lv6, rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        dense_feat = self.ASPP(dense_feat)  # Dense feature for lev 5
        # decoder 1 - Pyramid level 5
        lap_lv5 = torch.sigmoid(self.decoder1(dense_feat))
        lap_lv5_up = self.upscale(lap_lv5, scale_factor=2, mode='bilinear')

        # decoder 2 - Pyramid level 4
        dec2 = self.decoder2_up1(dense_feat)
        dec2 = self.decoder2_reduc1(torch.cat([dec2, cat3], dim=1))
        dec2_up = self.decoder2_1(torch.cat([dec2, lap_lv5_up, rgb_lv4], dim=1))
        dec2 = self.decoder2_2(dec2_up)
        dec2 = self.decoder2_3(dec2)
        lap_lv4 = torch.tanh(self.decoder2_4(dec2) + (0.1 * rgb_lv4.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian of image range is (-1,1)
        lap_lv4_up = self.upscale(lap_lv4, scale_factor=2, mode='bilinear')
        # decoder 2 - Pyramid level 3
        dec3 = self.decoder2_1_up2(dec2_up)
        dec3 = self.decoder2_1_reduc2(torch.cat([dec3, cat2], dim=1))
        dec3_up = self.decoder2_1_1(torch.cat([dec3, lap_lv4_up, rgb_lv3], dim=1))
        dec3 = self.decoder2_1_2(dec3_up)
        lap_lv3 = torch.tanh(self.decoder2_1_3(dec3) + (0.1 * rgb_lv3.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian of image range is (-1,1)
        lap_lv3_up = self.upscale(lap_lv3, scale_factor=2, mode='bilinear')
        # decoder 2 - Pyramid level 2
        dec4 = self.decoder2_1_1_up3(dec3_up)
        dec4 = self.decoder2_1_1_reduc3(torch.cat([dec4, cat1], dim=1))
        dec4_up = self.decoder2_1_1_1(torch.cat([dec4, lap_lv3_up, rgb_lv2], dim=1))

        lap_lv2 = torch.tanh(self.decoder2_1_1_2(dec4_up) + (0.1 * rgb_lv2.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian of image range is (-1,1)
        lap_lv2_up = self.upscale(lap_lv2, scale_factor=2, mode='bilinear')
        # decoder 2 - Pyramid level 1
        dec5 = self.decoder2_1_1_1_up4(dec4_up)
        dec5 = self.decoder2_1_1_1_1(torch.cat([dec5, lap_lv2_up, rgb_lv1], dim=1))
        dec5 = self.decoder2_1_1_1_2(dec5)
        lap_lv1 = torch.tanh(self.decoder2_1_1_1_3(dec5) + (0.1 * rgb_lv1.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian of image range is (-1,1)

        # Laplacian restoration
        lap_lv4_img = lap_lv4 + lap_lv5_up
        lap_lv3_img = lap_lv3 + self.upscale(lap_lv4_img, scale_factor=2, mode='bilinear')
        lap_lv2_img = lap_lv2 + self.upscale(lap_lv3_img, scale_factor=2, mode='bilinear')
        final_depth = lap_lv1 + self.upscale(lap_lv2_img, scale_factor=2, mode='bilinear')
        final_depth = torch.sigmoid(final_depth)
        return [(lap_lv5) * self.max_depth, (lap_lv4) * self.max_depth, (lap_lv3) * self.max_depth,
                (lap_lv2) * self.max_depth, (lap_lv1) * self.max_depth], final_depth * self.max_depth
        # fit laplacian image range (-80,80), depth image range(0,80)


class Lap_decoder_lv6(nn.Module):
    def __init__(self, args, dimList):
        super(Lap_decoder_lv6, self).__init__()
        norm = args.norm
        conv = conv_ws
        if norm == 'GN':
            if args.rank == 0:
                print("==> Norm: GN")
        else:
            if args.rank == 0:
                print("==> Norm: BN")

        if args.act == 'ELU':
            act = 'ELU'
        elif args.act == 'Mish':
            act = 'Mish'
        else:
            act = 'ReLU'
        kSize = 3
        self.max_depth = args.max_depth
        self.ASPP = Dilated_bottleNeck_lv6(norm, act, dimList[4])
        dimList[4] = dimList[4] // 2
        self.dimList = dimList
        ############################################     Pyramid Level 6     ###################################################
        # decoder1 out : 1 x H/32 x W/32 (Level 6)
        self.decoder1 = nn.Sequential(
            myConv(dimList[4] // 2, dimList[4] // 4, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[4] // 2) // 16),
            myConv(dimList[4] // 4, dimList[4] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[4] // 4) // 16),
            myConv(dimList[4] // 8, dimList[4] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[4] // 8) // 16),
            myConv(dimList[4] // 16, dimList[4] // 32, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[4] // 16) // 16),
            myConv(dimList[4] // 32, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[4] // 32) // 8)
            )
        ########################################################################################################################

        ############################################     Pyramid Level 5     ###################################################
        # decoder2 out : 1 x H/16 x W/16 (Level 5)
        # decoder2_up : (H/32,W/32)->(H/16,W/16)
        self.decoder2_up1 = upConvLayer(dimList[4] // 2, dimList[4] // 4, 2, norm, act, (dimList[4] // 2) // 16)
        self.decoder2_reduc1 = myConv(dimList[4] // 4 + dimList[3], dimList[4] // 4 - 4, kSize=1, stride=1, padding=0,
                                      bias=False,
                                      norm=norm, act=act, num_groups=(dimList[4] // 4 + dimList[3]) // 16)
        self.decoder2_1 = myConv(dimList[4] // 4, dimList[4] // 4, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[4] // 4) // 16)

        self.decoder2_2 = myConv(dimList[4] // 4, dimList[4] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[4] // 4) // 16)
        self.decoder2_3 = myConv(dimList[4] // 8, dimList[4] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[4] // 8) // 16)

        self.decoder2_4 = myConv(dimList[4] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[4] // 16) // 16)
        ########################################################################################################################

        ############################################     Pyramid Level 4     ###################################################
        # decoder2 out2 : 1 x H/8 x W/8 (Level 4)
        # decoder2_1_up2 : (H/16,W/16)->(H/8,W/8)
        self.decoder2_1_up2 = upConvLayer(dimList[4] // 4, dimList[4] // 8, 2, norm, act, (dimList[4] // 4) // 16)
        self.decoder2_1_reduc2 = myConv(dimList[4] // 8 + dimList[2], dimList[4] // 8 - 4, kSize=1, stride=1, padding=0,
                                        bias=False,
                                        norm=norm, act=act, num_groups=(dimList[4] // 8 + dimList[2]) // 16)
        self.decoder2_1_1 = myConv(dimList[4] // 8, dimList[4] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[4] // 8) // 16)

        self.decoder2_1_2 = myConv(dimList[4] // 8, dimList[4] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[4] // 8) // 16)

        self.decoder2_1_3 = myConv(dimList[4] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[4] // 16) // 16)
        ########################################################################################################################

        ############################################     Pyramid Level 3     ###################################################
        # decoder2 out3 : 1 x H/4 x W/4 (Level 3)
        # decoder2_1_1_up3 : (H/8,W/8)->(H/4,W/4)
        self.decoder2_1_1_up3 = upConvLayer(dimList[4] // 8, dimList[4] // 16, 2, norm, act, (dimList[4] // 8) // 16)
        self.decoder2_1_1_reduc3 = myConv(dimList[4] // 16 + dimList[1], dimList[4] // 16 - 4, kSize=1, stride=1,
                                          padding=0, bias=False,
                                          norm=norm, act=act, num_groups=(dimList[4] // 16 + dimList[1]) // 8)
        self.decoder2_1_1_1 = myConv(dimList[4] // 16, dimList[4] // 16, kSize, stride=1, padding=kSize // 2,
                                     bias=False,
                                     norm=norm, act=act, num_groups=(dimList[4] // 16) // 16)

        self.decoder2_1_1_2 = myConv(dimList[4] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                     norm=norm, act=act, num_groups=(dimList[4] // 16) // 16)
        ########################################################################################################################

        ############################################     Pyramid Level 2     ###################################################
        # decoder2 out4 : 1 x H/2 x W/2 (Level 2)
        # decoder2_1_1_1_up4 : (H/4,W/4)->(H/2,W/2)
        self.decoder2_1_1_1_up4 = upConvLayer(dimList[4] // 16, dimList[4] // 32, 2, norm, act,
                                              (dimList[4] // 16) // 16)
        self.decoder2_1_1_1_reduc4 = myConv(dimList[4] // 32 + dimList[0], dimList[4] // 32 - 4, kSize=1, stride=1,
                                            padding=0, bias=False,
                                            norm=norm, act=act, num_groups=(dimList[4] // 32 + dimList[0]) // 8)
        self.decoder2_1_1_1_1 = myConv(dimList[4] // 32, dimList[4] // 32, kSize, stride=1, padding=kSize // 2,
                                       bias=False,
                                       norm=norm, act=act, num_groups=(dimList[4] // 32) // 8)

        self.decoder2_1_1_1_2 = myConv(dimList[4] // 32, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                       norm=norm, act=act, num_groups=(dimList[4] // 32) // 8)
        ########################################################################################################################

        ############################################     Pyramid Level 1     ###################################################
        # decoder5 out : 1 x H x W (Level 1)
        # decoder2_1_1_1_1_up5 : (H/2,W/2)->(H,W)
        self.decoder2_1_1_1_1_up5 = upConvLayer(dimList[4] // 32, dimList[4] // 32 - 4, 2, norm, act,
                                                (dimList[4] // 32) // 8)  # H x W (64 -> 60)
        self.decoder2_1_1_1_1_1 = myConv(dimList[4] // 32, dimList[4] // 32, kSize, stride=1, padding=kSize // 2,
                                         bias=False,
                                         norm=norm, act=act, num_groups=(dimList[4] // 32) // 8)

        self.decoder2_1_1_1_1_2 = myConv(dimList[4] // 32, dimList[4] // 64, kSize, stride=1, padding=kSize // 2,
                                         bias=False,
                                         norm=norm, act=act, num_groups=(dimList[4] // 32) // 8)
        self.decoder2_1_1_1_1_3 = myConv(dimList[4] // 64, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                         norm=norm, act=act, num_groups=(dimList[4] // 64) // 4)
        ########################################################################################################################
        self.upscale = F.interpolate

    def forward(self, x, rgb):
        cat1, cat2, cat3, cat4, dense_feat = x[0], x[1], x[2], x[3], x[4]
        rgb_lv6, rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        dense_feat = self.ASPP(dense_feat)  # Dense feature for lev 6
        # decoder 1 - Pyramid level 6
        lap_lv6 = torch.sigmoid(self.decoder1(dense_feat))
        lap_lv6_up = self.upscale(lap_lv6, scale_factor=2, mode='bilinear')

        # decoder 2 - Pyramid level 5
        dec2 = self.decoder2_up1(dense_feat)
        dec2 = self.decoder2_reduc1(torch.cat([dec2, cat4], dim=1))
        dec2_up = self.decoder2_1(torch.cat([dec2, lap_lv6_up, rgb_lv5], dim=1))
        dec2 = self.decoder2_2(dec2_up)
        dec2 = self.decoder2_3(dec2)
        lap_lv5 = torch.tanh(self.decoder2_4(dec2) + (0.1 * rgb_lv5.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv5_up = self.upscale(lap_lv5, scale_factor=2, mode='bilinear')
        # decoder 2 - Pyramid level 4
        dec3 = self.decoder2_1_up2(dec2_up)
        dec3 = self.decoder2_1_reduc2(torch.cat([dec3, cat3], dim=1))
        dec3_up = self.decoder2_1_1(torch.cat([dec3, lap_lv5_up, rgb_lv4], dim=1))
        dec3 = self.decoder2_1_2(dec3_up)
        lap_lv4 = torch.tanh(self.decoder2_1_3(dec3) + (0.1 * rgb_lv4.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv4_up = self.upscale(lap_lv4, scale_factor=2, mode='bilinear')
        # decoder 2 - Pyramid level 3
        dec4 = self.decoder2_1_1_up3(dec3_up)
        dec4 = self.decoder2_1_1_reduc3(torch.cat([dec4, cat2], dim=1))
        dec4_up = self.decoder2_1_1_1(torch.cat([dec4, lap_lv4_up, rgb_lv3], dim=1))

        lap_lv3 = torch.tanh(self.decoder2_1_1_2(dec4_up) + (0.1 * rgb_lv3.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv3_up = self.upscale(lap_lv3, scale_factor=2, mode='bilinear')
        # decoder 2 - Pyramid level 2
        dec5 = self.decoder2_1_1_1_up4(dec4_up)
        dec5 = self.decoder2_1_1_1_reduc4(torch.cat([dec5, cat1], dim=1))
        dec5_up = self.decoder2_1_1_1_1(torch.cat([dec5, lap_lv3_up, rgb_lv2], dim=1))

        lap_lv2 = torch.tanh(self.decoder2_1_1_1_2(dec5_up) + (0.1 * rgb_lv2.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv2_up = self.upscale(lap_lv2, scale_factor=2, mode='bilinear')
        # decoder 2 - Pyramid level 1
        dec6 = self.decoder2_1_1_1_1_up5(dec5_up)
        dec6 = self.decoder2_1_1_1_1_1(torch.cat([dec6, lap_lv2_up, rgb_lv1], dim=1))
        dec6 = self.decoder2_1_1_1_1_2(dec6)
        lap_lv1 = torch.tanh(self.decoder2_1_1_1_1_3(dec6) + (0.1 * rgb_lv1.mean(dim=1, keepdim=True)))
        # if depth range is (0,1), laplacian image range is (-1,1)

        # Laplacian restoration
        lap_lv5_img = lap_lv5 + lap_lv6_up
        lap_lv4_img = lap_lv4 + self.upscale(lap_lv5_img, scale_factor=2, mode='bilinear')
        lap_lv3_img = lap_lv3 + self.upscale(lap_lv4_img, scale_factor=2, mode='bilinear')
        lap_lv2_img = lap_lv2 + self.upscale(lap_lv3_img, scale_factor=2, mode='bilinear')
        final_depth = lap_lv1 + self.upscale(lap_lv2_img, scale_factor=2, mode='bilinear')
        final_depth = torch.sigmoid(final_depth)
        return [(lap_lv6) * self.max_depth, (lap_lv5) * self.max_depth, (lap_lv4) * self.max_depth,
                (lap_lv3) * self.max_depth, (lap_lv2) * self.max_depth,
                (lap_lv1) * self.max_depth], final_depth * self.max_depth
        # fit laplacian image range (-80,80), depth image range(0,80)


# Laplacian Depth Residual Network
class LDRN(nn.Module):
    def __init__(self, args):
        super(LDRN, self).__init__()
        lv6 = args.lv6
        self.encoder = deepFeatureExtractor_ResNext101(args, lv6)

        if lv6 is True:
            self.decoder = Lap_decoder_lv6(args, self.encoder.dimList)
        else:
            self.decoder = Lap_decoder_lv5(args, self.encoder.dimList)

    def forward(self, x):
        out_featList = self.encoder(x)
        rgb_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        rgb_down4 = F.interpolate(rgb_down2, scale_factor=0.5, mode='bilinear')
        rgb_down8 = F.interpolate(rgb_down4, scale_factor=0.5, mode='bilinear')
        rgb_down16 = F.interpolate(rgb_down8, scale_factor=0.5, mode='bilinear')
        rgb_down32 = F.interpolate(rgb_down16, scale_factor=0.5, mode='bilinear')
        rgb_up16 = F.interpolate(rgb_down32, rgb_down16.shape[2:], mode='bilinear')
        rgb_up8 = F.interpolate(rgb_down16, rgb_down8.shape[2:], mode='bilinear')
        rgb_up4 = F.interpolate(rgb_down8, rgb_down4.shape[2:], mode='bilinear')
        rgb_up2 = F.interpolate(rgb_down4, rgb_down2.shape[2:], mode='bilinear')
        rgb_up = F.interpolate(rgb_down2, x.shape[2:], mode='bilinear')
        lap1 = x - rgb_up
        lap2 = rgb_down2 - rgb_up2
        lap3 = rgb_down4 - rgb_up4
        lap4 = rgb_down8 - rgb_up8
        lap5 = rgb_down16 - rgb_up16
        rgb_list = [rgb_down32, lap5, lap4, lap3, lap2, lap1]

        d_res_list, depth = self.decoder(out_featList, rgb_list)
        return d_res_list, depth

    def train(self, mode=True):
        super().train(mode)
        self.encoder.freeze_bn()
