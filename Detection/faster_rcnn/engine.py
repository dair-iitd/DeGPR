import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import cv2
import numpy as np
from sklearn import mixture
from skimage.feature import canny
from skimage.feature import hog
import torchvision


def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    if(abs(gmm_p.weights_[1]) < 1e-12 ):
        gmm_p.weights_[0] = 1.
        gmm_p.weights_[1] = 0.

    #print(gmm_p.weights_)
    X,y = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

def calc_postreg_loss_gmm(train_sample, test_sample,gmm_comp):
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

def Compute_shape_loss(device, embed_classifier, target_shape, pred_shape, target_shape_label):
    '''
    Compute loss between Ground truth and predicted Label for shape embedding
    '''
    print(np.shape(pred_shape))
    pred_shape = check_shape(target_shape, pred_shape)
    #check if both the target and predictions are same
    target_shape = flatten_list(target_shape)
    pred_shape = flatten_list(pred_shape)
    print(np.shape(target_shape), np.shape(pred_shape), np.shape(target_shape_label))
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    target_shape_embeddings =  torch.Tensor(target_shape)
    pred_shape_embeddings = torch.Tensor(pred_shape)
    target_shape_label = torch.Tensor(target_shape_label)
    print(pred_shape_embeddings.size(), target_shape_embeddings.size(), target_shape_label.size())
    total_loss = 0.0
    # get predictions
    output = embed_classifier(pred_shape_embeddings.to(device))
    loss = criterion(output, target_shape_label)
    print(loss)
    # for pred_shape_embedding in enumerate(pred_shape_embeddings):
    #     output = embed_classifier(pred_shape_embedding)
    #     loss = loss(output, target_shape_label)
    #     total_loss +=loss
    # total_loss = total_loss/len(targte_shape_label)
    return total_loss

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


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None,postreg=False,gmm_comp=2,gmm_version=1,num_classes=2,shape_model=None,shape_transform=None,embeddings_pca_model=None,embed_classifier=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    cpu_device = torch.device('cpu')

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            #losses = sum(loss for loss in loss_dict.values())

            # posterior regularisation starts
            #-----------------------------------

            if postreg:
                model.eval()
                outputs = model(images)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

                # initialise intensity lists
                target_intensity = [[] for _ in range(num_classes)]
                pred_intensity = [[] for _ in range(num_classes)]

                # initialise size lists
                target_size = [[] for _ in range(num_classes)]
                pred_size = [[] for _ in range(num_classes)]

                # initialise shape lists
                target_shape = [[] for _ in range(num_classes)]
                pred_shape = [[] for _ in range(num_classes)]

                target_shape_label = []
                pred_shape_label = []

                for i in range(len(images)):
                    img = torchvision.transforms.ToPILImage()(images[i])
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

                    target = targets[i]

                    intensity = [0 for _ in range(num_classes)]
                    size = [0 for _ in range(num_classes)]
                    shape = [np.zeros(512) for _ in range(num_classes)]
                    iel_embed_list = []
                    epith_embed_list = []
                    num_each_class = [0 for _ in range(num_classes)]

                    for j in range(len(target["labels"])):
                        label = target["labels"][j]
                        if(label == 0):
                            continue
                        box = target["boxes"][j]

                        x1 , y1, x2, y2 = int(box[0]) , int(box[1]) , int(box[2]) , int(box[3])

                        pixel_sum = np.sum(img[y1:y2,x1:x2,2])
                        pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))

                        if shape_model is not None:
                            try:
                                box_img = cv2.cvtColor(img[y1:y2, x1:x2],cv2.COLOR_HSV2RGB)
                                box_img = shape_transform(box_img)
                                box_img = box_img.unsqueeze(0)
                                box_img = box_img.to(device)
                                embed = shape_model.encoder(box_img)
                                embed = embed.squeeze(0)
                                embed = embed.detach().cpu().numpy()
                                target_shape_label.append(int(label))
                                # shape[int(labelsn[j][0])] += embed
                                # shape[j] = embed
                                if label==2:
                                    iel_embed_list.append(embed)
                                else:
                                    epith_embed_list.append(embed)
                                box_img = box_img.detach().cpu()
                                del box_img
                            except:
                                pass

                        intensity[label] += pixel_sum
                        size[label] += (x2 - x1 + 1)*(y2 - y1 + 1)
                        num_each_class[label] += 1

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

                    shape[2] = iel_embed_array
                    shape[1] = epith_embed_array

                    for k in range(num_classes):
                        if(num_each_class[k] != 0):
                            intensity[k] /= num_each_class[k]
                            size[k] /= num_each_class[k]

                        target_intensity[k].append(intensity[k])
                        target_size[k].append(size[k])
                        target_shape[k] = shape[k]

                    output = outputs[i]

                    intensity = [0 for _ in range(num_classes)]
                    size = [0 for _ in range(num_classes)]
                    shape = [np.zeros(512) for _ in range(num_classes)]
                    num_each_class = [0 for _ in range(num_classes)]
                    iel_embed_list = []
                    epith_embed_list = []

                    for j in range(len(output["labels"])):
                        label = output["labels"][j]
                        if(label == 0):
                            continue
                        box = output["boxes"][j]

                        x1 , y1, x2, y2 = int(box[0]) , int(box[1]) , int(box[2]) , int(box[3])

                        pixel_sum = np.sum(img[y1:y2,x1:x2,2])
                        pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))

                        if shape_model is not None:
                            try:
                                box_img = cv2.cvtColor(img[y1:y2, x1:x2],cv2.COLOR_HSV2RGB)
                                box_img = shape_transform(box_img)
                                box_img = box_img.unsqueeze(0)
                                box_img = box_img.to(device)
                                embed = shape_model.encoder(box_img)
                                embed = embed.squeeze(0)
                                embed = embed.detach().cpu().numpy()
                                pred_shape_label.append(label)
                                # shape[int(predn[j][5])] += embed
                                # shape[j] = embed
                                if label==2:
                                    iel_embed_list.append(embed)
                                else:
                                    epith_embed_list.append(embed)
                                box_img = box_img.detach().cpu()
                                del box_img
                            except:
                                pass

                        intensity[label] += pixel_sum
                        size[label] += (x2 - x1 + 1)*(y2 - y1 + 1)
                        num_each_class[label] += 1

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

                    shape[2] = iel_embed_array
                    shape[1] = epith_embed_array

                    for k in range(num_classes):
                        if(num_each_class[k] != 0):
                            intensity[k] /= num_each_class[k]
                            size[k] /= num_each_class[k]

                        pred_intensity[k].append(intensity[k])
                        pred_size[k].append(size[k])
                        pred_shape[k] = shape[k]
                # apply pca
                for i1 in range(num_classes):
                    try:
                        target_shape[i1] = np.array(target_shape[i1])
                        pred_shape[i1] = np.array(pred_shape[i1])
                        if gmm_comp!=4:
                            target_transform = embeddings_pca_model.transform(target_shape[i1].reshape(1,-1))
                            target_shape[i1] = target_transform.flatten()
                            pred_transform = embeddings_pca_model.transform(pred_shape[i1].reshape(1,-1))
                            pred_shape[i1] = pred_transform.flatten()
                    except:
                        print("pass")
                        pass

                lreg = 0
                num_pairs = 0
                for i1 in range(1,num_classes):
                    for i2 in range(i1+1,num_classes):
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

                            # print(target_shape1.shape, pred_shape1.shape)

                            pred_shape1 = pred_shape1.reshape((pred_shape1.shape[0],1))
                            target_shape1 = target_shape1.reshape((target_shape1.shape[0],1))

                            if shape_model is None:
                                lreg += calc_postreg_loss_gmm(target_intensity1 , pred_intensity1, gmm_comp)
                            else:
                                loss1 = calc_postreg_loss_gmm(np.concatenate((target_intensity1,target_size1)) , np.concatenate((pred_intensity1,pred_size1)), 2)
                                loss2 = calc_postreg_loss_gmm(target_shape1 , pred_shape1, 2)

                                print('Intensity loss : ' , loss1)
                                print('Embeddings loss :', loss2)
                                if loss2> 10000:
                                    lreg += (loss1)
                                else:
                                    lreg += (loss1 + 0.0001*loss2)

                    lreg /= num_pairs
                    loss_dict['posterior_loss'] = lreg

                model.train()
            #-----------------------------------
            # posterior regularisation ends

            losses = sum(loss for loss in loss_dict.values())


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
