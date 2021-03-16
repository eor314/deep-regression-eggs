import os
import glob
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset as BaseDataset


class PlanktonRegressionDataset(BaseDataset):
    """
    PlanktonDataset. Read images from list in VOC format, apply augmentation and preprocessing transformations.
    :param root: root of VOC-like directory [str]
    :param img_set: path to image list or list to consider [str OR list] 
    :param transform: transforms to preform for augmentations [func]
    :return: dataset object
    """
        
    def __init__(self, root=None, img_set=None, transform=None):

        self.root = root
        img_dir = os.path.join(root, 'JPEGImages')
        seg_dir = os.path.join(root, 'SegmentationMask')
        
        # if the image set given is not an absolute path, assume it lives in VOC structure
        if isinstance(img_set, list):
            # if the input is list of ids, put them in the proper place
            self.ids = img_set
        else:
            # otherwise read the file
            if not os.path.isabs(img_set):
                img_set = os.path.join(root, 'ImageSets', 'Main', img_set)

            # read and split the list assuming each line reads 
            #     img-id %egg
            
            # get the list
            with open(img_set, 'r') as ff:
                tmp = list(ff)
                ff.close
            tmp = [line.strip() for line in tmp]
            self.ids = [line.split(' ')[0] for line in tmp]
            self.pcts = [np.float32(line.split(' ')[1]) for line in tmp]
        
        self.images = [os.path.join(img_dir, f'{line}.jpg') for line in self.ids]
        
        self.transform = transform
            
    def __getitem__(self, ii):
        
        # read data
        with open(self.images[ii], 'rb') as ff:
            image = Image.open(ff)
            image = image.convert('RGB')
            ff.close()

        
        # apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.pcts[ii]
        
        # label is returned with a dummy dimension for the loss function
        return image, np.array([label])
    
    def __len__(self):
        return len(self.ids)
    
