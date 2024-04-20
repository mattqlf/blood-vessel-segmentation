from torch.utils.data import Dataset
import numpy as np
import cv2
from rembg import remove
from skimage.exposure import match_histograms
from config import Config

class init_load(Dataset):
    def __init__(self, paths, ref = False, bg = True):
        self.paths = paths
        self.ref = ref
        self.bg = bg
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[np.newaxis, ...]
        if not self.bg:
            x = remove(x)
            x = cv2.cvtColor(x, cv2.COLOR_RGBA2GRAY)[np.newaxis, ...]
        if self.ref:
            x = match_histograms(x[np.newaxis, ...], Config.ref[np.newaxis, ...], channel_axis=0)
        return x.squeeze(), path
    
class final_load(Dataset):
    def __init__(self, list_x, list_y, slices = 1, augmentations = None, norm = None, clip = None, infer = False):
        self.list_x = list_x
        self.list_y = list_y
        self.slices = slices
        self.augmentations = augmentations
        self.norm = norm
        self.clip = clip
        self.infer = infer
    
    def __len__(self):
        return sum([len(x) for x in self.list_x])
    
    def __getitem__(self, idx):
        j = 0
        for x in self.list_x:
            if idx>len(x)-self.slices:
                idx-=len(x)
                j+=1
            else:
                break
        x = self.list_x[j]
        y = self.list_y[j]
        
        img = np.stack([x[idx+i][0] for i in range(self.slices)], axis=0)
        label = np.stack([y[idx+i][0] for i in range(self.slices)], axis=0)
        path = '/'.join(x[idx][1].split('/')[-3:])
        
        if not self.infer:
            padded_img, padded_label = [], []
            for i in range(self.slices):
                image = img[i]
                mask = label[i]

                if image.shape[0] < Config.image_size:
                    if (Config.image_size - image.shape[0]) % 2 == 0:
                        top, bottom = int((Config.image_size - image.shape[0])/2), int((Config.image_size - image.shape[0])/2)
                    else:
                        top, bottom = (Config.image_size - image.shape[0])//2, (Config.image_size - image.shape[0])//2 + 1
                    image = cv2.copyMakeBorder(image, top = top, bottom = bottom, left = 0, right = 0, borderType = cv2.BORDER_REFLECT)
                    mask = cv2.copyMakeBorder(mask, top = top, bottom = bottom, left = 0, right = 0, borderType = cv2.BORDER_REFLECT)
                if image.shape[1] < Config.image_size:
                    if (Config.image_size - image.shape[1]) % 2 == 0:
                        left, right = int((Config.image_size - image.shape[1])/2), int((Config.image_size - image.shape[1])/2)
                    else:
                        left, right = (Config.image_size - image.shape[1])//2, (Config.image_size - image.shape[1])//2 + 1
                    image = cv2.copyMakeBorder(image, top = 0, bottom = 0, left = left, right = right, borderType = cv2.BORDER_REFLECT)
                    mask = cv2.copyMakeBorder(mask, top = 0, bottom = 0, left = left, right = right, borderType = cv2.BORDER_REFLECT)

                if image.shape[0] > Config.image_size:
                    start = (image.shape[0] - Config.image_size)//2
                    image = image[start : start + Config.image_size, :]
                    mask = mask[start : start + Config.image_size, :]

                if image.shape[1] > Config.image_size:
                    start = (image.shape[1] - Config.image_size)//2
                    image = image[:, start : start + Config.image_size]
                    mask = mask[:, start : start + Config.image_size]

                padded_img.append(image), padded_label.append(mask)       
            img, label = np.stack(padded_img), np.stack(padded_label)
        
        if self.augmentations:
            augmented = self.augmentations(image = img, mask = label)
            img = augmented['image']/255
            label = augmented['mask']/255
            
        if self.norm:
            norm = self.norm
            img = (img - norm[0])/(norm[1])
            if self.clip:
                ub, lb = self.clip
                img[img > ub] = (img[img>ub] - ub)*1e-3 + ub
                img[img < lb] = (img[img<lb] - lb)*1e-3 + lb
                
        return img.permute(1,2,0), label.permute(1,2,0), path