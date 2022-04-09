import torch
import torch.nn as nn

class SOAConv(nn.Module):

    def __init__(self, nf, k_size=3):
        super(SOAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class GCN(nn.Module):

    def __init__(self, c, out_c, k=(5,5) ):

        super(GCN, self).__init__()

        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding = [ int((k[0]-1)/2),0] )
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding = [0,int((k[0]-1)/2)])

        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k[1]), padding = [0,int((k[1]-1)/2)] )
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding = [int((k[1]-1)/2),0] )
        self.activation = nn.LeakyReLU(negative_slope=0.1,inplace=True)

    def forward(self,x):

        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r
        x = self.activation(x)
        return x

class SOAME(nn.Module):

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SOAME, self).__init__()

        group_width = nf // reduction
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Sequential(
            nn.Conv2d(group_width // 2 * 3, group_width, kernel_size=3, stride=stride,padding=dilation,dilation=dilation,bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride,padding=dilation,dilation=dilation, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.gcn_kernel7  = GCN(group_width, group_width // 2, (7, 7))
        self.gcn_kernel11 = GCN(group_width, group_width // 2, (11, 11))
        self.gcn_kernel15 = GCN(group_width, group_width // 2, (15, 15))

        self.SOA = SOAConv(group_width)
        self.conv3 = nn.Conv2d(group_width * reduction, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self,x):

        residual = x

        #split
        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        #ME
        out_a_gcn7 = self.gcn_kernel7(out_a)
        out_a_gcn11 = self.gcn_kernel11(out_a)
        out_a_gcn15 = self.gcn_kernel15(out_a)
        out_a = torch.cat((out_a_gcn7,out_a_gcn11,out_a_gcn15),1)
        out_a = self.k1(out_a)

        #SOA
        out_b = self.SOA(out_b)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a,out_b],dim=1))
        out += residual

        return out