#involution(out_channels, 7, 1)
#fpn_conv = involution(out_channels, 7, 1)
 
 #########https://github.com/d-li14/involution/blob/main/det/mmdet/models/utils/involution_naive.py
'''
 
    Example:

        >>> import torch

        >>> in_channels = [2, 3, 5, 7]

        >>> scales = [340, 170, 84, 43]

        >>> inputs = [torch.rand(1, c, s, s)

        ...           for c, s in zip(in_channels, scales)]

        >>> self = FPN(in_channels, 11, len(in_channels)).eval()

        >>> outputs = self.forward(inputs)

        >>> for i in range(len(outputs)):

        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')

        outputs[0].shape = torch.Size([1, 11, 340, 340])

        outputs[1].shape = torch.Size([1, 11, 170, 170])

        outputs[2].shape = torch.Size([1, 11, 84, 84])

        outputs[3].shape = torch.Size([1, 11, 43, 43])

'''
import torch.nn as nn
import torch

#from mmcv.cnn import ConvModule
#from models.common import Conv
###########
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)#
        #self.conv = DOConv2d(c1, c2, kernel_size=k, stride=s, padding =autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act =  nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
        
        

class involution(nn.Module):



    def __init__(self,

                 channels,

                 kernel_size,

                 stride):

        super(involution, self).__init__()

        self.kernel_size = kernel_size

        self.stride = stride

        self.channels = channels

        reduction_ratio = 4

        self.group_channels = 16

        self.groups = self.channels // self.group_channels
        
        
        self.bn = nn.BatchNorm2d(channels)  # applied to cat(cv2, cv3)
        self.act = nn.ReLU()
        self.conv1=Conv(

            channels, #in_channels=channels,

            channels // reduction_ratio, #out_channels=channels // reduction_ratio,

            k=1,
            )
        self.conv2=nn.Conv2d(

            channels // reduction_ratio, #in_channels=channels // reduction_ratio,

            kernel_size**2 * self.groups, #out_channels=kernel_size**2 * self.groups,

            1,  #kernel_size=1

            1,  #stride=1
            )
       
            
        '''
        self.conv1 = ConvModule(

            in_channels=channels,

            out_channels=channels // reduction_ratio,

            kernel_size=1,

            conv_cfg=None,

            norm_cfg=dict(type='BN'),

            act_cfg=dict(type='ReLU'))

        self.conv2 = ConvModule(

            in_channels=channels // reduction_ratio,

            out_channels=kernel_size**2 * self.groups,

            kernel_size=1,

            stride=1,

            conv_cfg=None,

            norm_cfg=None,

            act_cfg=None)
        '''

        if stride > 1:

            self.avgpool = nn.AvgPool2d(stride, stride)

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)



    def forward(self, x):

        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))

        b, c, h, w = weight.shape
       

        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)

        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)

        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)

        return self.act(self.bn(out)) #out
        
        
class Bottleneck_involution(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck_involution, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = involution(c_, 7, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class Bottleneck_involution_CSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(Bottleneck_involution_CSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)  ###################
        self.m = nn.Sequential(*[Bottleneck_involution(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))



if __name__ == '__main__':
    t = torch.ones((5, 512, 32,32))
    #sk0 = Conv(512,7,1)
    #sk0 = Bottleneck_involution(512,512)
    sk0 = Bottleneck_involution_CSP(512,256)    
    output = sk0(t)
    print(output.size())
