import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import cv2
import os
import time
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
cuda = torch.cuda.is_available()
from torchvision import datasets
from torchvision import transforms
import matplotlib as mpl
from dataset import SiameseCeliac
import argparse
from network import SupConResNet
from utils import TwoCropTransform, AverageMeter, TwoCropTransform_random_crop
from utils import adjust_learning_rate, warmup_learning_rate
from utils import set_optimizer, save_model
from losses import SupConLoss
import torch.backends.cudnn as cudnn
import sys
import Linear_classifier
from dataset import make_weights_for_balanced_classes, make_weights_for_balanced_classes_2_1
import random
import math

train_classifier = True
#file_paths celiac disease
directory_path_root = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold4/'
train_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold4/train/'
train_directory_path_2 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_10P/kfold4/train/'
train_directory_path_3 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_20P/kfold4/train/'
train_directory_path_1 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_5P/kfold4/train/'
train_directory_path_4 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_shifted_boxes_10P/kfold4/train/'
val_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold4/val/'

# # # file path for ConSep
# directory_path_root = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches/'
# train_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches/train/'
# train_directory_path_1 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches_5P/train/'
# train_directory_path_2 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches_10P/train/'
# train_directory_path_3 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches_20P/train/'
# val_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches/test/'

# file path for MonuSac
# directory_path_root = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MonuSac_cropped_patches/train/'
# train_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MonuSac_cropped_patches/train/'
# train_directory_path_1 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MoNuSac_cropped_patches_5P/train/'
# train_directory_path_2 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MoNuSac_cropped_patches_10P/train/'
# train_directory_path_3 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MoNuSac_cropped_patches_20P/train/'
# val_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MonuSac_cropped_patches/test/'

def parser_option(config):
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=config.epochs,
                        help='number of training epochs')
    parser.add_argument('--linear_epochs', type = int, default=config.linear_epochs,
                        help = 'number of training epochs for linear classifier')

    # optimization
    parser.add_argument('--SupCon_learning_rate', type=float, default= config.lr,
                        help='learning rate')
    parser.add_argument('--learning_rate', type=float, default = config.linear_lr,
                        help = 'linear classifier learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=config.momentum,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default= config.architecture,
                        choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--dataset', type=str, default=config.dataset,
                        choices=['path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--image_size', type = int, default = config.image_size, help = 'image size')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=config.temp,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_Balance_Inc_dec_shift_models_ResNet101{}'.format(opt.dataset, config.kfold)
    opt.tb_path = './save/SupCon/{}_Balance_Inc_dec_shift_tensorboard_ResNet101{}'.format(opt.dataset, config.kfold)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_kfold_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.SupCon_learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, config.kfold)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.SupCon_learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.SupCon_learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.SupCon_learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)


    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'Celiac_Disease':
        opt.n_cls =  2
    elif opt.dataset == 'MonuSac':
        opt.n_cls = 4
    elif opt.dataset == 'ConSep':
        opt.n_cls = 3
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    return opt


def get_balance_data(directory_path, transform):
    dataset_train = datasets.ImageFolder(directory_path, transform=TwoCropTransform(transform))
    if config.class_imbalance:
        weights = make_weights_for_balanced_classes_2_1(dataset_train.imgs, len(dataset_train.classes))
    else:
        weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle = False, sampler = sampler, num_workers=24)

    return data_loader

def get_balance_data_crop(directory_path, train_transform, train_crop_transform):
    dataset_train = datasets.ImageFolder(directory_path, transform=TwoCropTransform_random_crop(train_transform, train_crop_transform))
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle = False, sampler = sampler, num_workers=24)

    return data_loader

def get_loader(opt):
    # load datasets
    # Data loader for inference
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
                                    transforms.Resize(config.image_size),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])
    train_crop_transform = transforms.Compose([
                                    transforms.RandomCrop(size = (32,32)),
                                    transforms.Resize(config.image_size),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])


    test_transform = transforms.Compose([
                                    transforms.Resize(config.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])



    if config.inc_dec and config.balance:
        print("Inc_Dec and balance")
        train_data = datasets.ImageFolder(train_directory_path, transform=TwoCropTransform(train_transform))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle= True)
        #5%
        train_loader_5 = get_balance_data(train_directory_path_1, train_transform)
        #10%
        train_loader_10 = get_balance_data(train_directory_path_2, train_transform)
        #20%
        train_loader_20 = get_balance_data(train_directory_path_3, train_transform)
        #10% shift
        train_loader_shift_10 = get_balance_data(train_directory_path_4, train_transform)

        valid_data = datasets.ImageFolder(val_directory_path, transform=TwoCropTransform(test_transform))
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle = False)

        return train_loader, train_loader_5, train_loader_10, train_loader_20, train_loader_shift_10, valid_loader

    else:
        train_data = datasets.ImageFolder(train_directory_path, transform=TwoCropTransform(train_transform))
        valid_data = datasets.ImageFolder(val_directory_path, transform=TwoCropTransform(test_transform))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle= True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle = False)

        return train_loader, valid_loader


def ViT_model():
    model = ViT(image_size = 256, patch_size = 16, num_classes = 2, dim = 1024, depth = 6, heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1)
    return model

def set_model(opt):
    if opt.model =='ViT':
        print("Loading ViT Model")
        model = ViT_model()
    else:
        model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()


    return losses.avg


def valid(valid_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(valid_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(valid_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()


    return losses.avg


def train_linear_classifier(opt):
    best_acc = 0
    # build data loader
    train_loader, val_loader, test_loader = Linear_classifier.set_loader(opt)
    # build model and criterion
    model, classifier, criterion = Linear_classifier.set_model(opt)
    # build optimizer
    optimizer = Linear_classifier.set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.linear_epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = Linear_classifier.train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = Linear_classifier.validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))

    loss, test_acc, f1, precision, recall = Linear_classifier.test(test_loader, model, classifier, criterion, opt)
    print("Test: accuracy:{}, F1:{}, Precision:{}, Recall:{}".format(test_acc, f1, precision, recall))

    return classifier, loss, test_acc, f1, precision, recall

#load models

def main(config):

    opt = parser_option(config)

    # build data loader
    if config.inc_dec:
        train_loader, train_loader_5, train_loader_10, train_loader_20, train_loader_shift_10,valid_loader = get_loader(opt)
    else:
        train_loader, valid_loader = get_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.SupCon_learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        #randomly select the dataset
        if config.inc_dec:
            if epoch < 50:
                print("Phase 1")
                loss = train(train_loader, model, criterion, optimizer, epoch, opt)
            if 50 <= epoch <=100:
                print("Phase 2")
                n = random.randint(0,5)
                if n==0:
                    loss = train(train_loader_5, model, criterion, optimizer, epoch, opt)
                if n==1:
                    loss = train(train_loader_10, model, criterion, optimizer, epoch, opt)
                if n==2:
                    loss = train(train_loader_20, model, criterion, optimizer, epoch, opt)
                if n==3:
                    loss = train(train_loader_shift_10, model, criterion, optimizer, epoch, opt)
                else:
                    loss = train(train_loader, model, criterion, optimizer, epoch, opt)
            if epoch >100:
                print("Phase 3")
                n = random.randint(0,4)
                if n==0:
                    loss = train(train_loader_5, model, criterion, optimizer, epoch, opt)
                if n==1:
                    loss = train(train_loader_10, model, criterion, optimizer, epoch, opt)
                if n==2:
                    loss = train(train_loader_20, model, criterion, optimizer, epoch, opt)
                if n==3:
                    loss = train(train_loader_shift_10, model, criterion, optimizer, epoch, opt)

        else:
            loss = train(train_loader, model, criterion, optimizer, epoch, opt)

        val_loss = valid(valid_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        wandb.log({"Train Loss": loss, 'Valid Loss': val_loss})


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_{}_epoch_{}_temp_{}.pth'.format(config.architecture, epoch, config.temp))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    if train_classifier:
        opt.ckpt = save_file
        classifier, loss, test_acc, f1, precision, recall = train_linear_classifier(opt)
        print("Test accuracy:{}".format(test_acc))
        wandb.log({'Test Accuracy': test_acc, 'F1_score':f1, 'Precision': precision, 'Recall':recall, 'Test Loss':loss})

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'classifier_last_{}.pth'.format(config.kfold))
        save_model(classifier, optimizer, opt, opt.epochs, save_file)


if __name__ =='__main__':
    main(config)
