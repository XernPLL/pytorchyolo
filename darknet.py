from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

cfgfile = "cfg/yolov3.cfg"

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def load_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n') # list lines
    lines = [x for x in lines if len(x) > 0] # remove empty lines
    lines = [x for x in lines if x[0] != "#"] # remove commented lines
    lines = [x.rstrip().lstrip() for x in lines] # remove whitespaces on both sides

    block = {}
    blocks = [] # init list with dir of each block

    for line in lines:
        if line[0] == "[":
            if len(block) != 0: # dont do on first iter
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip().lstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0] # on first postion in cfg is general info about net
    module_list = nn.ModuleList() # list of nn.Module objects
    prev_filters = 3 # init filters, depth (RGB - 3)
    output_filters = [] # to track number of filters for each layer

    for idx, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type'] == "convolutional":
            activation = x['activation']
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False # bc normalized, we dont need to add another bias
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x["pad"])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size-1) // 2 # input and output will have the same spatial dimensions
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(idx),conv)

            if batch_normalize: # add batch normalization layer
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(idx), bn)

            if activation == "leaky": # add activation layer
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(idx), activn)

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = stride, mode = "bilinear")
            module.add_module("upsample_{0}".format(idx), upsample)

        elif x["type"] == "route":
            x["layers"] = x["layers"].split(',')

            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx

            route = EmptyLayer()
            module.add_module("route_{0}".format(idx), route)

            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                filters = output_filters[idx + start]


        elif x["type"] == "shortcut": # skip connection
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(idx), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(idx), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return(net_info,module_list)

blocks = load_cfg(cfgfile)
x,y = create_modules(blocks)
print(y)