# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image

#from GPUtil import showUtilization as gpu_usage

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh, check_version, xywh2xyxy)
import cv2
import numpy as np
import sys

from sklearn import mixture
import torchvision


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    if(abs(gmm_p.weights_[1]) < 1e-12 ):
        gmm_p.weights_[0] = 1.
        gmm_p.weights_[1] = 0.

    # print(gmm_p.weights_)
    X,y = gmm_p.sample(n_samples)
    # X_hat, y_hat = gmm_q.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

def calc_postreg_loss_gmm(train_sample, test_sample):
    if(train_sample.shape[0] < 2):
        return 0
    g1 = mixture.GaussianMixture(n_components=2,random_state=0).fit(train_sample)
    g2 = mixture.GaussianMixture(n_components=2,random_state=0).fit(test_sample)

    if(abs(g1.weights_[0] - 1) < 1e-15):
        g1.weights_[0] = 1.
        g1.weights_[1] = 0.
    if(abs(g2.weights_[0] - 1) < 1e-15):
        g2.weights_[0] = 1.
        g2.weights_[1] = 0.

    return gmm_kl(g1,g2)

class compute_shape_loss(nn.Module):
    def __init__(self, embed_classifier):
        super(compute_shape_loss, self).__init__()
        # define criteria
        self.crossEntloss = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.embed_classifier = embed_classifier

    def forward(self, device, pred_embed, pred_label,criteria = 'mse'):
        '''
        compute loss between predicited embedding and predicted label
        '''
        pred_embed = torch.Tensor(pred_embed)
        pred_label = torch.Tensor(pred_label)
        pred_label = pred_label.to(device)
        output = self.embed_classifier(pred_embed.to(device))
        if criteria=='mse':
            target_value = output.softmax(dim=0)
            target_label = torch.argmax(target_value)
            loss = self.mse(pred_label, target_label)
        else:
            pass
        return loss


# def compute_shape_loss(device, embed_classifier, pred_embed, pred_label):
#     '''
#     Load classifier and predict the class for embedding
#     Compute loss based on predicted yolo and predicted classifier label
#     '''
#     print(pred_embed)


def flatten_list(_2d_list):
    '''
    convert list of list into list
    flatten_list
    '''
    flat_list = []

    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)

    return flat_list

def check_shape(target_shape, pred_shape):
    pred_shape_new = []
    for i, pred_list in enumerate(pred_shape):
        print(np.shape(pred_list), np.shape(target_shape[i]))
        if np.shape(pred_list) == np.shape(target_shape[i]):
            pred_shape_new.append(pred_list)

    return pred_shape_new


class ComputeLoss:
    # Compute losses
    def __init__(self, model, embed_classifer = None, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride', 'no':
            setattr(self, k, getattr(det, k))

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        if embed_classifer is not None:
            self.compute_shape_loss = compute_shape_loss(embed_classifer)
        else:
            pass

    def __call__(self, p, targets,images=None,paths=None,gmm_reg=0.1,shape_model=None,shape_transform=None,embeddings_pca_model=None):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lreg = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        loss1_final = torch.zeros(1, device=device)
        loss2_final  = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        size_list_iel = []
        size_list_epith = []
        intensity_list_iel = []
        intensity_list_epith = []
        shape_list_iel = []
        shape_list_epith = []
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        #print("Before Compute loss")
        #print(gpu_usage())
        # posterior regularisation starts
        # -------------------------------------------------------------
        if (images is not None) or (paths is not None):
            imgs = []
            if images is not None:
                for j in range(images.shape[0]):
                    img = torchvision.transforms.ToPILImage()(images[j])
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                    imgs.append(img)

            if paths is not None:
                for j in range(len(paths)):
                    img = cv2.imread(paths[j])
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                    imgs.append(img)

            out = []

            for j in range(self.nl):
                bs, _, ny, nx, _ = p[j].shape
                if self.grid[j].shape[2:4] != p[j].shape[2:4]:
                    self.grid[j], self.anchor_grid[j] = self._make_grid(nx, ny, j)

                y = p[j].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[j]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh

                y = y.detach().cpu()

                out.append(y.view(bs, -1, self.no))

            out = non_max_suppression(torch.cat(out,1), 0.25, 0.6, labels=[], multi_label=True, agnostic=False)

            # initialise intensity lists
            target_intensity = [[] for _ in range(self.nc)]
            pred_intensity = [[] for _ in range(self.nc)]

            # initialise size lists
            target_size = [[] for _ in range(self.nc)]
            pred_size = [[] for _ in range(self.nc)]

            # initialise shape lists
            target_shape = [[] for _ in range(self.nc)]
            pred_shape = [[] for _ in range(self.nc)]

            target_shape_label = []
            pred_shape_label = []

            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class

                predn = pred.clone()

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    # labels -> [class,x1,y1,x2,y2] , predn -> [x1,y1,x2,y2,conf,class]

                    #Get information about ground truth

                    intensity = [0 for _ in range(self.nc)]
                    size = [0 for _ in range(self.nc)]
                    shape = [np.zeros(512) for _ in range(self.nc)]
                    # shape = [np.zeros(512) for _ in range(labelsn.shape[0])]
                    iel_embed_list = []
                    epith_embed_list = []
                    num_each_class = [0 for _ in range(self.nc)]

                    for j in range(labelsn.shape[0]):
                        x1 , y1 , x2, y2 = int(labelsn[j][1]*imgs[si].shape[1]) , int(labelsn[j][2]*imgs[si].shape[0]) , int(labelsn[j][3]*imgs[si].shape[1]) , int(labelsn[j][4]*imgs[si].shape[0])
                        pixel_sum = np.sum(imgs[si][y1:y2,x1:x2,2])
                        pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))
                        pixel_size_check = (x2 - x1 + 1)*(y2 - y1 + 1)
                        if int(labelsn[j][0])==0:
                            intensity_list_iel.append((pixel_sum, labelsn[j][0].detach().cpu().numpy()))
                            size_list_iel.append((pixel_size_check, labelsn[j][0].detach().cpu().numpy()))
                        else:
                            intensity_list_epith.append((pixel_sum, labelsn[j][0].detach().cpu().numpy()))
                            size_list_epith.append((pixel_size_check, labelsn[j][0].detach().cpu().numpy()))

                        if shape_model is not None:
                            try:
                                box_img = cv2.cvtColor(imgs[si][y1:y2, x1:x2],cv2.COLOR_HSV2RGB)
                                box_img = shape_transform(box_img)
                                box_img = box_img.unsqueeze(0)
                                box_img = box_img.to(device)
                                embed = shape_model.encoder(box_img)
                                embed = embed.squeeze(0)
                                embed = embed.detach().cpu().numpy()
                                target_shape_label.append(int(labelsn[j][0]))
                                
                                if int(labelsn[j][0])==0:
                                    iel_embed_list.append(embed)
                                    shape_list_iel.append((embed, labelsn[j][0].detach().cpu().numpy()))
                                else:
                                    epith_embed_list.append(embed)
                                    shape_list_epith.append((embed, labelsn[j][0].detach().cpu().numpy()))
                                box_img = box_img.detach().cpu()
                                del box_img
                            except:
                                print("except shape model")
                                pass

                        #calculate mean of IEL and Epith
                        intensity[int(labelsn[j][0])] += pixel_sum
                        size[int(labelsn[j][0])] += (x2 - x1 + 1)*(y2 - y1 + 1)
                        num_each_class[int(labelsn[j][0])] += 1

                    iel_embed_array = np.array(iel_embed_list)
                    epith_embed_array = np.array(epith_embed_list)
                    if len(iel_embed_array)!=0:
                        iel_embed_array = np.mean(iel_embed_array, axis = 0)
                    else:
                        iel_embed_array = np.zeros(512)
                    if len(epith_embed_array)!=0:
                        epith_embed_array = np.mean(epith_embed_array, axis = 0)
                    else:
                        epith_embed_array = np.zeros(512)

                    shape[0] = iel_embed_array
                    shape[1] = epith_embed_array


                    for k in range(self.nc):
                        if(num_each_class[k] != 0):
                            intensity[k] /= num_each_class[k]
                            size[k] /= num_each_class[k]

                        target_intensity[k].append(intensity[k])
                        target_size[k].append(size[k])
                        target_shape[k] = shape[k]

                    # Get information about Predictions

                    intensity = [0 for _ in range(self.nc)]
                    size = [0 for _ in range(self.nc)]
                    shape = [np.zeros(512) for _ in range(self.nc)]
                    num_each_class = [0 for _ in range(self.nc)]
                    iel_embed_list = []
                    epith_embed_list = []
                    for j in range(predn.shape[0]):
                        x1 , y1 , x2, y2 = int(predn[j][0]) , int(predn[j][1]) , int(predn[j][2]) , int(predn[j][3])
                        x1 , y1 , x2 , y2 = max(x1,0) , max(y1,0) , max(x2,0) , max(y2,0)

                        pixel_sum = np.sum(imgs[si][y1:y2,x1:x2,2])
                        pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))
                        if shape_model is not None:
                            try:
                                box_img = cv2.cvtColor(imgs[si][y1:y2, x1:x2],cv2.COLOR_HSV2RGB)
                                box_img = shape_transform(box_img)
                                box_img = box_img.unsqueeze(0)
                                box_img = box_img.to(device)
                                embed = shape_model.encoder(box_img)
                                embed = embed.squeeze(0)
                                embed = embed.detach().cpu().numpy()
                                pred_shape_label.append(int(predn[j][5]))

                                if int(predn[j][5])==0:
                                    iel_embed_list.append(embed)
                                else:
                                    epith_embed_list.append(embed)
                                box_img = box_img.detach().cpu()
                                del box_img
                            except:
                                pass


                        intensity[int(predn[j][5])] += pixel_sum
                        size[int(predn[j][5])] += (x2 - x1 + 1)*(y2 - y1 + 1)
                        num_each_class[int(predn[j][5])] += 1

                    iel_embed_array = np.array(iel_embed_list)
                    epith_embed_array = np.array(epith_embed_list)
                    if len(iel_embed_array)!=0:
                        iel_embed_array = np.mean(iel_embed_array, axis = 0)
                    else:
                        iel_embed_array = np.zeros(512)
                    if len(epith_embed_array)!=0:
                        epith_embed_array = np.mean(epith_embed_array, axis = 0)
                    else:
                        epith_embed_array = np.zeros(512)

                    shape[0] = iel_embed_array
                    shape[1] = epith_embed_array
                    total_patches = 0
                    for k in range(self.nc):
                        if(num_each_class[k] != 0):
                            intensity[k] /= num_each_class[k]
                            size[k] /= num_each_class[k]
                            total_patches +=num_each_class[k]

                        pred_intensity[k].append(intensity[k])
                        pred_size[k].append(size[k])
                        pred_shape[k] = shape[k]

            for l in range(len(out)):
                out[l] = out[l].detach().cpu()
            del out
            del imgs

            # apply pca
            for i1 in range(self.nc):
                try:
                    target_shape[i1] = np.array(target_shape[i1])
                    pred_shape[i1] = np.array(pred_shape[i1])
                    target_transform = embeddings_pca_model.transform(target_shape[i1].reshape(1,-1))
                    target_shape[i1] = target_transform.flatten()
                    pred_transform = embeddings_pca_model.transform(pred_shape[i1].reshape(1,-1))
                    pred_shape[i1] = pred_transform.flatten()
                except:
                    # print("Pass")
                    pass

            num_pairs = 0
            for i1 in range(self.nc):
                for i2 in range(i1+1,self.nc):
                    num_pairs += 1
                    target_intensity1 = [np.subtract(x1, x2) for (x1, x2) in zip(target_intensity[i1], target_intensity[i2])]
                    pred_intensity1 = [np.subtract(x1, x2) for (x1, x2) in zip(pred_intensity[i1], pred_intensity[i2])]

                    if len(target_intensity1) > 0 and len(pred_intensity1) > 0:
                        target_intensity1 = np.array(target_intensity1)
                        target_intensity1 = target_intensity1.reshape((target_intensity1.shape[0],1))

                        target_size1 = [np.subtract(x1, x2) for (x1, x2) in zip(target_size[i1], target_size[i2])]
                        target_size1 = np.array(target_size1)
                        target_size1 = target_size1.reshape((target_size1.shape[0],1))
                        pred_intensity1 = np.array(pred_intensity1)
                        pred_intensity1 = pred_intensity1.reshape((pred_intensity1.shape[0],1))

                        pred_size1 = [np.subtract(x1, x2) for (x1, x2) in zip(pred_size[i1], pred_size[i2])]
                        pred_size1 = np.array(pred_size1)
                        pred_size1 = pred_size1.reshape((pred_size1.shape[0],1))

                        target_shape1 = np.subtract(target_shape[i1] , target_shape[i2])
                        pred_shape1 = np.subtract(pred_shape[i1] , pred_shape[i2])

                        pred_shape1 = pred_shape1.reshape((pred_shape1.shape[0],1))
                        target_shape1 = target_shape1.reshape((target_shape1.shape[0],1))

                        if shape_model is None:
                            lreg += calc_postreg_loss_gmm(np.concatenate((target_intensity1,target_size1)) , np.concatenate((pred_intensity1,pred_size1)))
                        else:
                            loss1 = calc_postreg_loss_gmm(np.concatenate((target_intensity1,target_size1)) , np.concatenate((pred_intensity1,pred_size1)))
                            loss2 = calc_postreg_loss_gmm(target_shape1 , pred_shape1)

                            print('Intensity loss : ' , loss1)
                            print('Embeddings loss :', loss2)
                    
                            if loss1> 100:
                                if loss2>100:
                                    lreg+=(0.0001* min(loss1, loss2))
                                    loss1_final += 0.0001*loss1
                                    loss2_final += 0.0001*loss2
                                else:
                                    lreg += (loss2)
                                    loss1_final +=0.0001 * loss1
                                    loss2_final += loss2
                            if loss2>100 and loss1<100:
                                lreg+=loss1
                                loss1_final +=loss1
                                loss2_final += 0.0001* loss2
                            if loss2<100 and loss1<100:
                                lreg += (loss1 + 0.1*loss2)
                                loss1_final +=loss1
                                loss2_final += loss2
#                           
            lreg /= num_pairs
            loss1_final/= num_pairs
            loss2_final /=num_pairs

            print('Regularisation loss :' , lreg)
            final_loss = (lbox + lobj + lcls + lreg) * bs
            print_lreg = lreg
            lreg = lreg.detach().cpu()
            del lreg
            final_loss = (lbox + lobj + lcls) * bs
            return final_loss, torch.cat((lbox, lobj, lcls, print_lreg, loss1_final, loss2_final)).detach(), intensity_list_iel, intensity_list_epith, size_list_iel, size_list_epith, shape_list_iel, shape_list_epith

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls, lcls)).detach()

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
