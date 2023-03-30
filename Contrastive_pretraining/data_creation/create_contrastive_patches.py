import os
import numpy as np
import cv2
import PIL
import glob
import tqdm
import sys
from skimage.feature import canny
import matplotlib.pyplot as plt

# folder_path

# specify the directory where the entire images(on which we perform detection) are stored
image_file_path = '/home/aayush/chirag/open_src_datasets/consep/yolo_format/images/test/'

# specify the directory where the yolo labels corresponding to the images are stored
label_file_path = '/home/aayush/chirag/open_src_datasets/consep/yolo_format/labels/test'

# specify the directory where the contrastive training patches would be stored
image_save_path_root = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches/test/'

# specify the image size, typically 640 for celiac, 500 for consep and 320 for monusac
image_size = 500

# dataset types
#celiac_disase , ConSep, MoNuSac
dataset = 'ConSep'


# Get all files in folder
def convert_yolo_cart(numbers, width, height):
    coord = numbers[1:]
    coord[0] = coord[0] * height
    coord[1] = coord[1] * width
    coord[2] = coord[2] * height
    coord[3] = coord[3] * width
    xmin = int(coord[0] - coord[2]/2)
    ymin = int(coord[1] - coord[3]/2)
    xmax = int(coord[0] + coord[2]/2)
    ymax = int(coord[1] + coord[3]/2)
    if xmin <0:
        xmin = 0
    if ymin<0:
        ymin = 0
    if xmax> 640:
        xmax = 640
    if ymax> 640:
        ymax = 640

    return xmin,ymin,xmax,ymax

def converttoyolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def compute_edges_otsu(image):
    original = image.copy()
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
        break

    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    close = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(original, original, mask=close)
    result[close==0] = (255,255,255)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    return edges


def auto_canny(image, sigma=0.4):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def compute_edges(image):
    gray = cv2.bilateralFilter(image, 15, 75, 75)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = gray
#     edges = cv2.Canny(gray,100,200)
    return edges

def load_image(image_name):
    img = cv2.imread(image_name)
    img = cv2.resize(img, (640, 640))
    edge_img = compute_edges(img)
    return img, edge_img


def load_image_path_list(images_dir):
    image_file_path = images_dir
    image_file_path_list = []
    for image_file_name in glob.glob(os.path.join(image_file_path, '*.jpg')):
        image_file_path_list.append(image_file_name)

    return image_file_path_list

def crop_image(image, xmin, ymin, xmax, ymax):
    if xmin == xmax or ymin==ymax:
        sys.exit()
    else:
        cropped_image = image[ymin:ymax, xmin:xmax]
        cropped_image = cv2.resize(cropped_image, (64, 64))
        return cropped_image

def save_images(image_name, class_type, cropped_image, image_save_path_root, dataset):
    if dataset == 'celiac_disease':
        if class_type == 1.0:
            #save to IEL
            image_save_path = os.path.join(image_save_path_root, 'IEL/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
        else:
            #save in epith
            image_save_path = os.path.join(image_save_path_root, 'Epith/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
    if dataset == 'ConSep':
        if class_type ==0.0:
            #save to Inflammatory
            image_save_path = os.path.join(image_save_path_root, 'Inflammatory/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
        elif class_type == 1.0:
            #save to Epithelial
            image_save_path = os.path.join(image_save_path_root, 'Epithelial/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
        elif class_type == 2.0:
            #save to Spindle
            image_save_path = os.path.join(image_save_path_root, 'Spindle/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
            
    if dataset == 'MoNuSac':
        if class_type ==0.0:
            #save to Inflammatory
            image_save_path = os.path.join(image_save_path_root, 'Epithelial/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
        elif class_type == 1.0:
            #save to Epithelial
            image_save_path = os.path.join(image_save_path_root, 'Lymphocyte/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
        elif class_type == 2.0:
            #save to Spindle
            image_save_path = os.path.join(image_save_path_root, 'Neutrophil/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)
        elif class_type == 3.0:
            #save to Spindle
            image_save_path = os.path.join(image_save_path_root, 'Macrophage/{}'.format(image_name))
            cv2.imwrite(image_save_path, cropped_image)

def get_image_label(image_name, train_label_dir, edge = True, save = False):
   # read image
    img, edge_image = load_image(image_name)
    label_path = train_label_dir
    filename = os.path.split(image_name)[1]
    label_name = os.path.join(label_path, filename[:-4] + '.txt')
    if os.stat(label_name).st_size !=0:
        f = open(label_name, 'r')
        for i, line in enumerate(f):
            numbers = [float(num) for num in line.split()]
            xmin, ymin, xmax, ymax = convert_yolo_cart(numbers, image_size,image_size)
            if xmin != xmax and ymin != ymax:
                if edge:
                    cropped_image = crop_image(img, xmin, ymin, xmax, ymax)
                    cropped_image = compute_edges_otsu(cropped_image)
                else:
                    cropped_image = crop_image(img, xmin, ymin, xmax, ymax)
                image_name  = filename[:-4] + '_{}'.format(i) + '.jpg'
                if cv2.countNonZero(edge_image) != 0:
                    if save:
                        save_images(image_name, numbers[0], cropped_image, image_save_path_root, dataset)

    return img, edge_image



image_path_list = load_image_path_list(image_file_path)
print("found :{} files".format(len(image_path_list)))
for image_path in tqdm.tqdm(image_path_list):
    img, edge_image = get_image_label(image_path, label_file_path, False, True)