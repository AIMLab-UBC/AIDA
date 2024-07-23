import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from augmentation.CutOut import CutOut
from augmentation.SizeJitter import SizeJitter
import matplotlib.pyplot as plt

class PatchDataset(Dataset):
    def __init__(self, x_set, y_set, CategoryEnum, state=None, resize=None, fft_enhancer=False, target_paths=None, training_set=False):

        """
        Args:
            x_set (string): List of paths to images
            y_set (int): Labels of each image in x_set
            # resize (int): if we want to resize the image first
            training_set: Whether training dataset or not
        """
        self.x_set = x_set
        self.y_set = y_set
        self.training_set = training_set
        self.resize = resize
        self.fft_enhancer = fft_enhancer
        self.target_paths = target_paths
        self.state = state

        self.transform = self.get_transform()
        self.length = len(x_set)
        self.classes_(CategoryEnum)
        self.ratio_()
        if len(x_set) != len(y_set):
            raise ValueError('x set length does not match y set length')

    def get_transform(self):
        transforms_array = []
        length_cut = int(self.original_size() * 0.2)
        if self.resize is not None:
            transforms_array.append(transforms.Resize(self.resize))
            length_cut = int(self.resize * 0.2)
        if self.state == 'external_train':
            transforms_array.extend([transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.5),
                                     SizeJitter(ratio=0.1, prob=0.3, color="black"),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     CutOut(num_cut=1, length_cut=length_cut, color="black")])

        elif self.training_set:
            if self.fft_enhancer:
                transforms_array.append(swap_amplitude(self.target_paths))
            transforms_array.extend([transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.5),
                                     SizeJitter(ratio=0.1, prob=0.3, color="black"),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     CutOut(num_cut=1, length_cut=length_cut, color="black")])
        else:
            transforms_array.append(transforms.ToTensor())
        transforms_array.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        transforms_ = transforms.Compose(transforms_array)
        return transforms_

    def ratio_(self):
        ''' Find the ratio of each class compare to others
        useful for balancing
        it should be 1-real_ratio
        '''
        _, ratio = np.unique(self.y_set, return_counts=True)
        ratio = 1 / (ratio / len(self.y_set))
        self.ratio = [ratio_/sum(ratio) for ratio_ in ratio]

    def classes_(self, CategoryEnum):
        self.classes = []
        for y_ in set(self.y_set):
            self.classes.append(CategoryEnum(y_).name)

    def original_size(self):
        x = Image.open(self.x_set[0][0]).convert('RGB')
        return transforms.ToTensor()(x).shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.x_set[idx][0]
        x = Image.open(path).convert('RGB')
        y = self.y_set[idx]
        x = self.transform(x)
        slide_id = self.x_set[idx][1]
        (x_,  y_) = os.path.splitext(os.path.basename(path))[0].split('_')
        (x_,  y_) = (int(x_),  int(y_))
        return x, torch.tensor(y), slide_id, (x_,y_)

class swap_amplitude(object):
    
    def __init__(self, target_paths):
        self.target_paths = target_paths

    def __call__(self, source_img):
        p = random.uniform(0, 1)
        if p < 0.6:
            return source_img

        random.shuffle(self.target_paths)
            
        self.rand_idx = random.randint(0, len(self.target_paths)-1)
        self.target_img_path = self.target_paths[self.rand_idx]
        
        x = np.array(source_img).astype(np.uint8)
        target_img = Image.open(self.target_img_path).convert('RGB')
        y = np.array(target_img).astype(np.uint8)

        im1 = Image.fromarray(x.copy())
        im2 = Image.fromarray(y.copy())

        fft_x = np.fft.fftshift(np.fft.fftn(x))
        fft_y = np.fft.fftshift(np.fft.fftn(y))

        abs_x, angle_x = np.abs(fft_x), np.angle(fft_x)
        abs_y, angle_y = np.abs(fft_y), np.angle(fft_y)

        fft_x = abs_y * np.exp((1j) * angle_x)

        x = np.fft.ifftn(np.fft.ifftshift(fft_x))
        x = x.astype(np.uint8)

        x = Image.fromarray(x)
        
        return x

    def __repr__(self):
        return self.__class__.__name__+'()'
        
