from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


weightsfile = "yolov3.weights" # PATH TO WEIGHTS FILE
cfgfile = "cfg/yolov3.cfg"
impath = "in_images/dog-cycle-car.png"
outpath = "out_images/frame.jpg"
batch_size = 1
confidence = 0.5 # obcject confidence to filter predictions
nms_thesh = 0.4 # NMS threshold
reso = "416" # input resolution of the network (320, 416, 608)


CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")
# Set up the neural network
print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print("Network successfully loaded")

model.net_info["height"] = reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

# Detection phase

loaded_ims = cv2.imread(impath)

img = prep_image(loaded_ims, inp_dim)
im_dim = loaded_ims.shape[1], loaded_ims.shape[0]
im_dim = torch.FloatTensor(im_dim).repeat(1, 2)



if CUDA:
    im_dim = im_dim.cuda()
    im_batches = img.cuda()


with torch.no_grad():
    prediction = model(Variable(img), CUDA)

prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)


output = prediction

try:
    output
except NameError:
    print("No detections were made")
    exit()

im_dim = im_dim.repeat(output.size(0), 1)
scaling_factor = torch.min(int(reso) / im_dim, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img

list(map(lambda x: write(x, loaded_ims), output))


cv2.imwrite(outpath, loaded_ims)

key = cv2.waitKey(0)


torch.cuda.empty_cache()