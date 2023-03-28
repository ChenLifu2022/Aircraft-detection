""" PyConv networks for image recognition as presented in our paper:
    Duta et al. "Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"
    https://arxiv.org/pdf/2006.11538.pdf
"""
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


class PyConv2d(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``

    Example::

        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):

    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
        
class PyConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2,pyconv_kernels,pyconv_groups, stride=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(PyConv, self).__init__()
        
        self.conv = get_pyconv(c1, c2, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


##############
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



    
class PyConv_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2,pyconv_kernels, pyconv_groups,shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(PyConv_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        #self.cv2 = Conv(c_, c2, 3, 1, g=g)
        
        self.cv2 = PyConv(c1=c_, c2=c2, pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups,stride=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class Pyconv_BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True,pyconv_kernels=[3, 5,7,9], pyconv_groups=[1, 4,8,16],e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(Pyconv_BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[PyConv_Bottleneck(c_, c_,pyconv_kernels, pyconv_groups, shortcut, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
 ###############   
 


class Fusion_up_GCAM(nn.Module):

    def __init__(self,
                 
                 in_channels,
                 out_channels,
                 ):

        super(Fusion_up_GCAM, self).__init__()
        all_channels =0
        for j in range(3-1):
            all_channels =all_channels + in_channels[j]
        print(all_channels)            
        self.conv = nn.Conv2d(all_channels, out_channels, 1)

        #self.bn = nn.BatchNorm2d(out_channels)
        #self.relu =nn.ReLU(inplace=True)

        self.m=Pyconv_BottleneckCSP(c1=512,c2=512,shortcut=False,pyconv_kernels=[3,5],pyconv_groups=[1,4])  #add

    def forward(self, inputs):

        #Residual Feature Augmentation

        h, w = inputs[-1].size(2), inputs[-1].size(3)
        channels =0
        #Ratio Invariant Adaptive Pooling
        AdapPool_Features = [F.interpolate(inputs[j], size=(h,w), mode='nearest') for j in range(len(inputs)-1)] 
            #channels +=(inputs[j].size(1))
            #print('111',AdapPool_Features.shape)
        #print('111',AdapPool_Features[0].shape)
        #print('222',AdapPool_Features[1].shape)
        AdapPool_Features[0] = self.m(AdapPool_Features[0])
        AdapPool_Features = [AdapPool_Features[0],AdapPool_Features[1]]


        Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
        #print('fff',Concat_AdapPool_Features.shape)
        out_Features =self.conv(Concat_AdapPool_Features) 
        
        #out_Features = self.bn(out_Features)
        #out_Features = self.relu(out_Features)
        
        return out_Features  #此处的inputs[0]是主干输入特征图没有经过1x1卷积调整通道后地最顶层特征图




if __name__ == "__main__":
    #m=PyConv(c1=64, c2=32, pyconv_kernels=[3, 5], pyconv_groups=[1, 4],stride=1,)
    m=Pyconv_BottleneckCSP(32,32,False,[3,5,7],[1,4,8])
    input = torch.randn(1, 32, 32, 32)
    output = m(input)
    print('Pyconv_BottleneckCSP:',output.shape)
    ####################传参yolov5中BottleneckCSP（64，False,[3,5,7],[1,4,8]）####################
        
    #m=PyConv(c1=64, c2=32, pyconv_kernels=[3, 5], pyconv_groups=[1, 4],stride=1,)
    #m=BottleneckCSP(64,64,False,[3,5,7],[1,4,8])
    input0 = torch.randn(4, 512, 16, 16)
    input = torch.randn(4, 256, 32, 32)
    input1 = torch.randn(4, 128, 64, 64)
    model = Fusion_up_GCAM([512,256,128],128)

    down_stre= Conv(128, 128, 3, 2)
    down_stre111 =down_stre(input1)

    output = model([input0,input,down_stre111])    

    print('222',output.shape)
    print('333',down_stre111.shape)
    output1 = torch.cat([output,down_stre111], dim=1)
    print('1444',output1.shape)

    #model1 = Add(2)
    #output = model1([input0,input1])

