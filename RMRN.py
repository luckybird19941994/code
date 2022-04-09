import torch
import torch.nn as nn
import torch.nn.functional as F
import queue
import copy
from torch.nn import init
import functools

class GCN(nn.Module):

    def __init__(self, c, out_c, k=(5,5) ):

        super(GCN, self).__init__()

        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding = [ int((k[0]-1)/2),0] )
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding = [0,int((k[0]-1)/2)])

        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k[1]), padding = [0,int((k[1]-1)/2)] )
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding = [int((k[1]-1)/2),0] )
        self.activation = nn.ReLU(inplace=True)

    def forward(self,x):

        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r
        x = self.activation(x)
        return x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):

    def __init__(self, nf=64):

        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class neuro(nn.Module):
    def __init__(self,n_c=64,n_b=10):
        super(neuro, self).__init__()

        self.conv_1 = nn.Conv2d(n_c+3+3*3, n_c, kernel_size=3, stride=1, padding=1)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_trunk = make_layer(basic_block,n_b)
        self.conv_h = nn.Conv2d(n_c*2, n_c, kernel_size=3, stride=1, padding=1)
        self.conv_o = nn.Conv2d(n_c*2, 3, kernel_size=3, stride=1, padding=1)
        initialize_weights([self.conv_1,self.conv_h,self.conv_o], 0.1)

        self.conv_ori = nn.Conv2d(n_c,n_c,kernel_size=1,stride=1)
        self.gcn_kernel11 = GCN(n_c, 32, (11, 11))
        self.gcn_kernel15 = GCN(n_c, 32, (15, 15))

    def forward(self,x,h,o):
        x = torch.cat((x,h,o),dim=1)
        x = F.relu(self.conv_1(x))
        x = self.recon_trunk(x)

        out_ori = self.conv_ori(x)
        out_11 = self.gcn_kernel11(x)
        out_15 = self.gcn_kernel15(x)
        x = torch.cat((out_11,out_15,out_ori),1)

        x_h = F.relu(self.conv_h(x))
        x_o = self.conv_o(x)
        return x_h, x_o

class RMRN(nn.Module):

    def __init__(self,n_c,n_b):
        super(RMRN, self).__init__()
        self.neuro = neuro(n_c,n_b)
        self.n_c = n_c

    def forward(self,f1,f2,recon_image,x_h,x_o):

        x_input = torch.cat((f1,f2,recon_image),dim=1)
        x_h,x_o = self.neuro(x_input,x_h,x_o)
        return x_h,x_o

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)






