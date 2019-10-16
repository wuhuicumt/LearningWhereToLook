import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class ChannelAttention(nn.Module):
    def __init__(self, in_dim):
        """
        Borrowed from https://github.com/sinashish/multi-scale-attention
        """
        super(ChannelAttention, self).__init__()
        self.in_dim = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        bs, c, h, w =x.size()

        proj_query = x.view(bs, c, -1)
        proj_key = x.view(bs, c, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = F.sigmoid(energy_new)
        proj_value = x.view(bs, c, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(bs, c, h, w)
        out = self.gamma*out +x

        return out

class LinearChannelAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearChannelAttention, self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.fc= nn.Sequential(
            nn.Linear(in_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, ch, h, w = x.size()

        y = self.gpool(x) 
        y = self.fc(y)
        
        out = self.gamma* y + x
        return out

class MultiAttentionSubnet(nn.Module):
    def __init__(self, in_ch):
        super(MultiAttentionSubnet, self).__init__()
        self.backbone = vgg19(pretrained=False).features
        
        self.gpool1 = nn.AdaptiveAvgPool2d(1)
        self.gpool2 = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(1000, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch//2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch//2),
            nn.ReLU(),
        )
        self.cam1 = ChannelAttention(in_ch)
        self.cam2 = ChannelAttention(in_ch)

        #Linear Activation 
        self.attn1 = LinearChannelAttention(1000, in_ch)
        self.attn2 = LinearChannelAttention(1000, in_ch)

    def forward(self, x):
        y = self.backbone(x)
        
        #y1 = self.gpool1(y)
        #y2 = self.gpool2(y)
        #y1 = self.cam1(y1)
        #y2 = self.cam2(y2)

        y1 = self.attn1(y)
        y2 = self.attn2(y)

        return y1, y2


class CroppingNetwork(nn.Module):
    def __init__(self, in_ch):
        super(CroppingNetwork, self)
        self.in_ch = in_ch
        self.fc = Sequential(
            nn.Linear(in_ch, in_ch//2)
            nn.ReLU(),
            nn.Linear(in_ch//2, in_ch//2),
            nn.ReLU()
        )

    def forward(self, attn_map, input_img):
        k = 10
        tx = torch.sum(attn_map, dim=1)
        ty = torch.sum(attn_map, dim=2)
        ts = torch.sum(attn_map, dim=3)
        func  = lambda x: 1/(1+torch.exp(k*x))
        vx = func(input_img - tx +0.5*ts) - func(input_img - tx - 0.5*tx)
        vy = func(input_img - tx + 0.5*ts) - func(input_img - ty - 0.5*ts)
        v = torch.dot(vx, vy)

        out = torch.mm(input_img, v)
        return out
        
class Model(nn.Module):
    def __init__(self, in_ch):
        super(Model, self).__init__()
        self.subnet = MultiAttentionSubnet(in_ch)
        self.crop1 = CroppingNetwork(in_ch//2)
        self.crop2 = CroppingNetwork(in_ch//2)
        self.backbone1 = vgg19(pretrained=False).features
        self.backbone2 = vgg19(pretrained=False).features
        self.backbone3 = vgg19(pretrained=False).features
        self.gpool1 = AdaptiveAvgPool2d(1)
        self.gpool2 = AdaptiveAvgPool2d(1)
        self.gpool3 = AdaptiveAvgPool2d(1)

    def forward(self, x):
        #MultiAttentionSubNet
        attn_maps = self.subnet(x)
        attn_map1, attn_map2 = attn_maps

        #Cropping SubNet
        fcnet1 = self.crop1(attn_map1, x)
        fcnet2 = self.crop2(attn_map2, x)

        #Joint Feature Learning
        #Global Features
        y1 = self.backbone1(x)
        y1 = self.gpool1(y1)

        #Local Features
        y2 = self.backbone2(fcnet1)
        y2 = self.gpool2(y2)
        y3 = self.backbone3(fcnet2)
        y3 = self.gpool3(y3)

        out = y1 + y2 + y3

        return out,\
            attn_map1,\
            attn_map2,\
            y1,\
            y2, \
            y3
    
            
            