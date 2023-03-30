# Basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from embeddings_model import get_embeddings_model, LinearClassifier, get_embedding_classifier
from torchvision import transforms
import pickle as pk

class IELDataset(torch.utils.data.Dataset):

    def __init__(self,images_dir, width, height,image_size=320,transforms=None):
        self.transforms = transforms
        self.images_dir = images_dir
        self.label_dir = images_dir.replace("images","labels")
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(images_dir))
                        if image[-4:]=='.png']

        for text_file in sorted(os.listdir(self.label_dir)):
            l = 0
            with open(os.path.join(self.label_dir,text_file),'r') as f:
                for x in f:
                    l += 1

            if l == 0:
                self.imgs.remove(text_file.replace('.txt','.png'))
        self.classes = ['background', 'Cell']
        # self.classes = ['background', 'Epithelial Nuclei','IEL']
        # self.classes = ['background', 'Inflammatory','Epithelial','Spindle']
        # self.classes = ['background', 'Epithelial','Lymphocyte','Neutrophil','Macrophage']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.images_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_CUBIC)
        # diving by 255
        img_res /= 255.0

        boxes = []
        labels = []

        wt = img.shape[1]
        ht = img.shape[0]

        label_path = os.path.join(self.label_dir, img_name.replace('.png','.txt'))

        with open(label_path,'r') as f:
            for line in f:
                splits = line.split(' ')
                w = float(splits[3]) * wt
                h = float(splits[4]) * ht
                x1 = ((2 * float(splits[1]) * wt) - w)/2
                y1 = ((2 * float(splits[2]) * ht) - h)/2
                x2 = x1 + w
                y2 = y1 + h

                x1 = max(0,(x1/wt)*self.width)
                x2 = min(self.width-1,(x2/wt)*self.width)
                y1 = max(0,(y1/ht)*self.height)
                y2 = min(self.height-1,(y2/ht)*self.height)

                if x1 >= x2 or y1 >= y2:
                    continue

                boxes.append([x1,y1,x2,y2])
                labels.append(int(splits[0]) + 1)

        boxes = [box for box in boxes if len(box) == 4]

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        if boxes.shape[0] == 0:
            area = boxes
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:

            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res , target

    def __len__(self):
        return len(self.imgs)

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):

    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #load model path
    # model_path = '/home/aayush/chirag/faster_rcnn/weights/faster_rcnn_pretrain_nuclei_seg/faster_rcnn_99fasterrcnn_pretrain_nuclei_seg.pt'
    # model = torch.load(model_path)


    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_data(opt):
    #fold_dir = "/home/aayush/chirag/open_src_datasets/monusac/yolo_format/images/"
    fold_dir = opt.data
    image_size = opt.imgsz

    # use our dataset and defined transformations
    dataset_train = IELDataset(os.path.join(fold_dir,"train"), image_size, image_size, transforms= get_transform(train=True))

    if(os.path.isdir(os.path.join(fold_dir,"val"))):
        dataset_val = IELDataset(os.path.join(fold_dir,"val"), image_size, image_size, transforms= get_transform(train=False))
    else:
        dataset_val = IELDataset(os.path.join(fold_dir,"test"), image_size, image_size, transforms= get_transform(train=False))

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=opt.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return data_loader_train, data_loader_val

def train_model(opt):
    num_epochs = opt.epochs
    num_classes = opt.num_classes + 1

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # get the model using our helper function
    model = get_object_detection_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    data_loader_train, data_loader_val = get_data(opt)

    # Initialize the embeddings model
    embed_model = None
    embed_transform = None
    pca_model = None
    embed_classifier = None
    if opt.embedding_weights != 'INVALID':
        embed_model , _ = get_embeddings_model(opt.embedding_weights)
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        embed_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, stds)])

        if opt.embedding_pca_weights != 'INVALID':
            pca_model = pk.load(open(opt.embedding_pca_weights,'rb'))
            print("PCA model loaded")

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10,postreg=opt.postreg,num_classes=num_classes, shape_model=embed_model, shape_transform=embed_transform, embeddings_pca_model=pca_model)  # loss scaled by batch_size)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)

        if (epoch+1)%25 ==0:
            torch.save(model,os.path.join('weights','faster_rcnn_{}'.format(epoch) + opt.save_file + '.pt'))

    #torch.save(model, './weights/faster-rcnn-monusac-postreg-v2-lambda1.pt')
    torch.save(model,os.path.join('weights','faster_rcnn_' + opt.save_file + '.pt'))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--data', type=str, default='INVALID', help='path to the images folder of data')
    parser.add_argument('--save_file', type=str, default='', help='name of the file where you want to store weights')
    parser.add_argument('--postreg', action='store_true', help='add posterior regularisation')
    parser.add_argument('--embedding_weights', type=str, default='INVALID', help='path to the weights of the embeddings model')
    parser.add_argument('--embedding_pca_weights', type=str, default='INVALID', help='path to the weights of the embeddings pca model')
    parser.add_argument('--warmup_postreg_epochs', type = int, default=0, help = 'number of warmup epochs before starting postreg loss')

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    train_model(opt)
