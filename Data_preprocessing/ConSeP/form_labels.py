import os
import cv2
import matplotlib.pyplot as plt
import scipy.io

import os
import numpy as np
import torch

import torchvision.transforms.functional as F
from PIL import Image
import torchvision.transforms as transforms

from planar import BoundingBox

root_dir = '/home/aayush/chirag/open_src_datasets/consep/CoNSeP/Test/'

img_paths = []

for subdir, dirs, files in os.walk(os.path.join(root_dir,'Images')):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".png"):
            img_paths.append(filepath)

for img_path in img_paths:
    label_path = img_path.replace('Images','Labels').replace('.png','.mat')

    img = cv2.imread(img_path)
    mat = scipy.io.loadmat(label_path)

    mask = mat['inst_map']
    class_types = mat['inst_type']

    bboxes = [] # class_type , xmin , ymin, xmax, ymax

    for i in range(class_types.shape[0]):
        temp = np.where(mask == (i+1))

        coords = [(x,y) for x,y in zip(temp[0],temp[1])]

        bbox = BoundingBox(coords)
        bboxes.append((int(class_types[i][0]),int(bbox.min_point[0]),int(bbox.min_point[1]),int(bbox.max_point[0]),int(bbox.max_point[1])))

    print_buffer = []
    for (clss,x1,y1,x2,y2) in bboxes:
        print_buffer.append("{} {} {} {} {}".format(clss,x1,y1,x2,y2))

    with open(img_path.replace('Images','Annotations').replace('.png','.txt'),'w') as f:
        f.write("\n".join(print_buffer))







