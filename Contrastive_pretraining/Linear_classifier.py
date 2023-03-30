from __future__ import print_function
import sys
import argparse
import time
import math
import os
import torch
import torch.backends.cudnn as cudnn
from utils import TwoCropTransform, AverageMeter
from utils import set_optimizer, save_model
import random
from utils import AverageMeter
from utils import adjust_learning_rate, warmup_learning_rate, accuracy, get_accuracy, precision_recall_f1score
from utils import set_optimizer
from network import SupConResNet, LinearClassifier
from torchvision import transforms, datasets
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from dataset import make_weights_for_balanced_classes


image_size = 224
batch_size = 8
inc_dec_balance = True

# #file_paths
# train_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold0/train/'
# val_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold0/val/'
# test_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold0/test/'

directory_path_root = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold0/'
train_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold0/train/'
train_directory_path_2 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_10P/kfold0/train/'
train_directory_path_3 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_20P/kfold0/train/'
train_directory_path_1 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_5P/kfold0/train/'
train_directory_path_4 = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold_shifted_boxes_10P/kfold0/train/'
val_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold0/val/'
test_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/Celiac_cropped_patches_kfold/kfold0/test/'
#
# #file_paths
# train_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches/train/'
# val_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches/test/'
# test_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/ConSep_cropped_patches/test/'


# train_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MonuSac_cropped_patches/train/'
# val_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MonuSac_cropped_patches/test/'
# test_directory_path = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Dataset/MonuSac_cropped_patches/test/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=25,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'ConSep', 'MonuSac','Celiac_Disease'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    parser.add_argument('--model_path', type=str, default = '',
                       help='path to save weights')
    parser.add_argument('--fold', type=str, default=0,
                        help='fold information')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_fold_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.fold)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'Celiac_Disease':
        print("Celiac_Disease")
        opt.n_cls =  2
    elif opt.dataset == 'MonuSac':
        print("MonuSac")
        opt.n_cls = 4
    elif opt.dataset == 'ConSep':
        print("ConSep")
        opt.n_cls = 3
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt




def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def get_balance_data(directory_path, transform):
    dataset_train = datasets.ImageFolder(directory_path, transform=transform)
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle = False, sampler = sampler, num_workers=24)

    return data_loader

def set_loader(opt):
    # load datasets
    # Data loader for inference
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])


    test_transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])

    train_data = datasets.ImageFolder(train_directory_path, transform=train_transform)
    valid_data = datasets.ImageFolder(val_directory_path, transform=test_transform)
    test_data = datasets.ImageFolder(test_directory_path, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle= True)
    val_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = False)

    return train_loader, val_loader, test_loader

def set_loader_inc_balance(opt):
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])
    test_transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])


    print("Inc_Dec and balance")
    # train_data = datasets.ImageFolder(train_directory_path, transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle= True)
    # #5%
    # train_loader_5 = get_balance_data(train_directory_path_1, train_transform)
    # #10%
    # train_loader_10 = get_balance_data(train_directory_path_2, train_transform)
    # #20%
    # train_loader_20 = get_balance_data(train_directory_path_3, train_transform)
    #
    # valid_data = datasets.ImageFolder(val_directory_path, transform=test_transform)
    # valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle = False)
    #
    # test_data = datasets.ImageFolder(test_directory_path, transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = False)


    train_data = datasets.ImageFolder(train_directory_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle= True)
    #5%
    train_loader_5 = get_balance_data(train_directory_path_1, train_transform)
    #10%
    train_loader_10 = get_balance_data(train_directory_path_2, train_transform)
    #20%
    train_loader_20 = get_balance_data(train_directory_path_3, train_transform)
    #10% shift
    train_loader_shift_10 = get_balance_data(train_directory_path_4, train_transform)

    valid_data = datasets.ImageFolder(val_directory_path, transform=test_transform)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle = False)

    test_data = datasets.ImageFolder(test_directory_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = False)


    return train_loader, train_loader_5, train_loader_10, train_loader_20, train_loader_shift_10, valid_loader, test_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        # acc1, acc5 = accuracy(output, labels, topk=(1,))
        acc = get_accuracy(output, labels)
        top1.update(acc, bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            # acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            acc = get_accuracy(output, labels)
            top1.update(acc, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def test(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            # acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            acc = get_accuracy(output, labels)
            top1.update(acc, bsz)
            f1_value, precision_value, recall_value = precision_recall_f1score(output, labels)
            f1.update(f1_value)
            precision.update(precision_value)
            recall.update(recall_value)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg, f1.avg, precision.avg, recall.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    if inc_dec_balance:
        train_loader, train_loader_5, train_loader_10, train_loader_20, train_loader_shift_10, val_loader, test_loader = set_loader_inc_balance(opt)
    else:
        train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if inc_dec_balance:
            n = random.randint(0,5)
            if n==0:
                loss, acc = train(train_loader_5, model, classifier, criterion, optimizer, epoch, opt)
            if n==1:
                loss, acc = train(train_loader_10, model, classifier, criterion, optimizer, epoch, opt)
            if n==2:
                loss, acc = train(train_loader_20, model, classifier, criterion, optimizer, epoch, opt)
            if n==3:
                loss, acc = train(train_loader_shift_10, model, classifier, criterion, optimizer, epoch, opt)
            else:
                loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        else:
            loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)

        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            save_file_best = os.path.join(
                opt.save_folder, 'ckpt_classifier_{}_epoch_{}_inc_def_balance_fold_{}_best.pth'.format(opt.model, epoch, opt.fold))
            save_model(classifier, optimizer, opt, epoch, save_file_best)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_classifier_{}_epoch_{}_inc_def_balance_fold_{}.pth'.format(opt.model, epoch, opt.fold))
            save_model(classifier, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last_classifier_inc_dec_balance_{}.pth'.format(opt.fold))
    save_model(classifier, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))

    loss, test_acc, f1, precision, recall = test(test_loader, model, classifier, criterion, opt)
    print("Test: accuracy:{}, f1:{}, precision:{}, recall:{}".format(test_acc, f1, precision, recall))


if __name__ == '__main__':
    main()
