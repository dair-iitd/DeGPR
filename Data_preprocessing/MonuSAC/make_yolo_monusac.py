import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import json
import cv2
import sys
from PIL import Image
import scipy.io as sio
import scipy
import scipy.ndimage
from skimage import draw
import xml.etree.ElementTree as ET
from planar import BoundingBox

root_dir = '/home/aayush/chirag/open_src_datasets/monusac/MoNuSAC Testing Data and Annotations/'
save_dir = '/home/aayush/chirag/open_src_datasets/monusac/yolo_format/images/test/'

labels = {'Epithelial' : 0 , 'Lymphocyte' : 1 , 'Neutrophil' : 2, 'Macrophage' : 3}

for root_dirs, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith('.tif'):
            img_file = os.path.join(root_dirs, filename)
            img = cv2.imread(img_file)
            iid = filename[:-4]

            y_ = img.shape[0]
            x_ = img.shape[1]

            (y1,x1) = (320 * round(img.shape[0] / 320) , 320 * round(img.shape[1] / 320))
            x1 , y1 = max(320,x1) , max(320,y1)

            x_scale = x1 / x_
            y_scale = y1 / y_

            img = cv2.resize(img , (y1,x1) , interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            bboxes = []
            label_path = img_file.replace('.tif','.xml')

            tree = ET.parse(label_path)
            root = tree.getroot()
                                    
            for k in range(len(root)):
                label = [x.attrib['Name'] for x in root[k][0]]
                label = label[0]
                if label not in labels:
                    continue
                fid = labels[label]
                
                for child in root[k]:
                    for x in child:
                        r = x.tag
                        
                        if r == 'Attribute':
                            label = x.attrib['Name']

                        if r == 'Region':
                            vertices = x[1]
                            coords = []
                            for i, vertex in enumerate(vertices):
                                x = float(vertex.attrib['X'])
                                y = float(vertex.attrib['Y'])
                                coords.append((x, y))
                            bbox = BoundingBox(coords)
                            bboxes.append((fid , int(bbox.min_point[0]),int(bbox.min_point[1]),int(bbox.max_point[0]),int(bbox.max_point[1])))

            from itertools import product

            w, h = img.size
            d = 320
            output_imgs = []
            k = 0

            print(w,h,d)
        
            grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
            for i, j in grid:
                box = (j, i, j+d, i+d)
                output_imgs.append(img.crop(box))

                output_imgs[-1].save(os.path.join(save_dir , iid + '_{}.jpg'.format(k)))

                print_buffer = []
                i , j = j , i

                for (fid,x1,y1,x2,y2) in bboxes:
                    if x1 >= x2 or y1 >= y2:
                        continue
                    x1 , y1 , x2 , y2 = int(x1 * x_scale) , int(y1 * y_scale) , int(x2 * x_scale) , int(y2 * y_scale)
                    b_center_x = (x1 + x2) / 2 
                    b_center_y = (y1 + y2) / 2
                    if (b_center_x > i and b_center_x < i+d) and (b_center_y > j and b_center_y < j+d):
                        b_width    = (x2 - x1)
                        b_height   = (y2 - y1)

                        b_center_x -= i
                        b_center_y -= j

                        b_center_x /= d
                        b_center_y /= d
                        b_width /= d
                        b_height /= d
                        
                        #Write the bbox details to the file 
                        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(fid, b_center_x, b_center_y, b_width, b_height))

                with open(os.path.join(save_dir.replace('images','labels') , iid + '_{}.txt'.format(k)) , 'w') as f:
                    f.write("\n".join(print_buffer))

                k += 1

print(labels)