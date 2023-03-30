import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import json
import cv2
import sys
from PIL import Image

root_dir = '/home/aayush/chirag/open_src_datasets/consep/CoNSeP/Train/'
save_dir = '/home/aayush/chirag/open_src_datasets/consep/yolo_format/images/train/'
img_size = 500

for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith('.png'):
            img_file = os.path.join(root, filename)
            img = cv2.imread(img_file)
            iid = filename[:-4]

            y_ = img.shape[0]
            x_ = img.shape[1]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            bboxes = []

            with open(os.path.join(root_dir , 'Annotations' , iid + '.txt') , 'r') as f:
                for line in f:
                    splits = line[:-1].split(' ')
                    fid,y1,x1,y2,x2 = int(splits[0]) , int(splits[1]), int(splits[2]), int(splits[3]), int(splits[4])

                    if fid == 2:
                        bboxes.append((0,x1,y1,x2,y2))
                    elif fid == 3 or fid == 4:
                        bboxes.append((1,x1,y1,x2,y2))
                    elif fid == 5 or fid == 6 or fid == 7:
                        bboxes.append((2,x1,y1,x2,y2))


            from itertools import product

            w, h = img.size
            d = img_size
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
                    b_center_x = (x1 + x2) / 2 
                    b_center_y = (y1 + y2) / 2
                    if (b_center_x > i and b_center_x < i+d) and (b_center_y > j and b_center_y < j+d):
                        b_width    = (x2 - x1)
                        b_height   = (y2 - y1)

                        b_center_x -= i
                        b_center_y -= j

                        b_center_x /= img_size
                        b_center_y /= img_size
                        b_width /= img_size
                        b_height /= img_size
                        
                        #Write the bbox details to the file 
                        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(fid, b_center_x, b_center_y, b_width, b_height))

                with open(os.path.join(save_dir.replace('images','labels') , iid + '_{}.txt'.format(k)) , 'w') as f:
                    f.write("\n".join(print_buffer))

                k += 1

