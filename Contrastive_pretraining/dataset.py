import numpy as np
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def make_weights_for_balanced_classes_2_1(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        if i ==0:
            weight_per_class[i] = 2 * N/float(count[i])
        else:
            weight_per_class[i] = 1 * N/float(count[i])
    weight = [0] * len(images)
    # print(weight_per_class)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

class Celiac_Hard_Negative(Dataset):
    '''
    Train: Create Hard Negative samples
    Test: Create Positive and Negative pairs
    '''
    def __init__(self, dataset_path, type, transforms):
        self.data_path = dataset_path
        self.train = type
        self.transform = transforms
        #get the labels files
        self.labels = []
        self.data_files = []
        self.labels_set = next(os.walk(self.data_path))[1]
        for label in self.labels:
            for file in glob.glob(os.path.join(self.data_path, label, '*.jpg')):
                self.data_files.append(file)
                self.labels.append(label)

class SiameseCeliac(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset_path, type, transforms):

        self.data_path =  dataset_path
        self.train = type
        self.transform = transforms

        self.labels = []
        self.data_files = []
        #get folder and labels
        self.labels_set  = next(os.walk(self.data_path))[1]
        for label in self.labels:
            for file in glob.glob(os.path.join(self.data_path, label, '*.jpg')):
                self.data_files.append(file)
                self.labels.append(label)

        self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                         for label in self.labels_set}