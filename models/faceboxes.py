import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(96, 24, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(96, 24, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(96, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 24, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(96, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(24, 24, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(24, 24, kernel_size=3, padding=1)
  
  def forward(self, x):
    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, ceil_mode=True)
    branch1x1 = self.branch1x1(branch1x1_pool)


    branch1x1_2 = self.branch1x1_2(x)
    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return torch.cat(outputs, 1)


class CRelu(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
  
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = F.relu(x, inplace=True)
    return x


class CRelu1(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu1, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
  
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x
    
    
class FaceBoxes(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(FaceBoxes, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size

    self.conv1_1 = CRelu(3, 16, kernel_size=5, stride=2, padding=2)
    self.conv1_2 = CRelu(16, 32, kernel_size=5, stride=2, padding=2)
    self.conv2 = CRelu(32, 96, kernel_size=3, stride=1, padding=1)
    
    self.inception1 = Inception()

    self.incep_2 = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)

    self.inception2 = Inception()

    self.conv3_2 = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)
    self.loc, self.conf = self.multibox(self.num_classes)
    
    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(96, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(96, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(96, 21 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(96, 21 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(96, 2 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(96, 2 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)
    
  def forward(self, x):
  
    sources = list()
    loc = list()
    conf = list()
    detection_dimension = list()

    x = self.conv1_1(x)
    # cc = np.squeeze(x.data).permute(2,1,0).contiguous()
    # with open("log.txt","w") as f:
    #   for i in range(cc.shape[0]):
    #     for j in range(cc.shape[1]):
    #       for k in range(cc.shape[2]):
      
        
    #         f.write(str(cc[i,j,k].tolist()))
    #         f.write(" ")
    #       f.write("\n")
    #     f.write("\n")
    x = self.conv1_2(x)

    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0,ceil_mode=True)

    x = self.conv2(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0,ceil_mode=True)
    x = self.inception1(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    x = self.incep_2(x)
    x = self.inception2(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    x = self.conv3_2(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)

    detection_dimension = torch.tensor(detection_dimension, device=x.device)

    for (x, l, c) in zip(sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)),
                detection_dimension)
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                detection_dimension)
  
    return output
