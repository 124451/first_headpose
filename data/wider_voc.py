import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import random
import numpy as np
from data.data_augment import preproc
import json


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import json

WIDER_CLASSES = ( '__background__', 'face')

my_img_dim = 256

class DLAnnotationTransform(object):

    def __call__(self, target):

        res = np.empty((0, 5))
        # assert len(target) % 5 == 0,"label is error!"

        for i in range(len(target)//5):
            bndbox = []
            x_mode = 0
            y_mode = 0
            if int(float(target[i*5])) < 0:
                bndbox.append(0)
                x_mode = 1
            else:
                bndbox.append(int(float(target[i*5])))
            if int(float(target[i*5 + 1])) < 0:
                y_mode = 1
                bndbox.append(0)
            else:
                bndbox.append(int(float(target[i*5 + 1])))

            if x_mode == 1:
                bndbox.append(int(float(target[i*5 + 2])) + int(float(target[i*5])))
            else:
                bndbox.append(int(float(target[i*5 + 2])))
            if y_mode == 1:
                bndbox.append(int(float(target[i*5 + 3])) + int(float(target[i*5 + 1])))
            else:
                bndbox.append(int(float(target[i*5 + 3])))
            bndbox.append(int(float(target[i*5 + 4])))
            res = np.vstack((res,bndbox))


        return res

def search(txtname):
    dst = {}
    lst = []
    with open(txtname,"r") as f:
        a = f.readlines()
        for line in a:
            m = line.strip().split(" ")
            lst.append(m[0])
            dst[m[0]] = m[1:]
    return dst,lst


class Negative_dataset(data.Dataset):
    def __init__(self, root,txtpath,ratio,preproc=None):
        self.root = root
        self.ratio = ratio
        self.preproc = preproc
        self.txtpath = txtpath
        self.dataauget = 1
        self._imgpath2 = self.root
        with open(self.txtpath, 'r') as f:
            self._imglist = [line.strip('\n') for line in f]
    def __len__(self):
        return len(self._imglist)


class VOCDetection(data.Dataset):

    def __init__(self, root,txtpath, ratio, preproc=None, target_transform=None):
        self.root = root
        
        self.ratio = ratio
        self.rescale_size = 1
        self.txtpath = txtpath
        self.dataauget = 0
      
        self.preproc = preproc
        self.target_transform = target_transform
        # self._annopath = os.path.join(self.root, 'annotations', '%s')
        self._imgpath = self.root
        self._imglist = list()
        with open(self.txtpath, 'r') as f:
          self._imglist = [line.strip("\n") for line in f]
        self.batch_count = 1
    def __len__(self):
        return len(self._imglist)

class Mixing_dataset(data.Dataset):
    def __init__(self,datasets):
        self.datasets = datasets
        self.rescale_size = 1.0
        self.batch_count = 0       

    def __getitem__(self, index):
        all_ratio=0.0
        flag = np.random.uniform(0.0, 1.0)
        # mydataset = 0
        # img_id = 0
        if flag < 0.1:
            mydataset = self.datasets[1]
            img_id = random.choice(mydataset._imglist)
        else:
            mydataset = self.datasets[0]
            img_id = random.choice(mydataset._imglist)
        # for dataset in self.datasets:
        #     a = all_ratio
        #     all_ratio += dataset.ratio
        #     if flag>=a and flag<=all_ratio:
        #         mydataset = dataset
        #         img_id = random.choice(dataset._imglist)
        valueLst = img_id.split(" ")
        # print("img_id:",img_id)

        if len(valueLst) < 3:
            img_path = img_id

            img = cv2.imread(os.path.join(mydataset._imgpath2, img_path), cv2.IMREAD_COLOR)
            if mydataset.preproc is not None:
                img = mydataset.preproc(img)

            return torch.from_numpy(img), np.array([[]])

        else:
            

            img = cv2.imread(os.path.join(mydataset._imgpath,valueLst[0]), cv2.IMREAD_COLOR)
            
            if mydataset.target_transform is not None:
                target = mydataset.target_transform(valueLst[1:])

            if mydataset.preproc is not None:
                mydataset.rescale_size = self.rescale_size
                mydataset.preproc.img_dim=(mydataset.rescale_size*my_img_dim)
                #print("preproc size",self.preproc.img_dim)
                img, target = mydataset.preproc(img, target)

            return torch.from_numpy(img), target

    def __len__(self):
        return len(self.datasets[0]._imglist)

    def detection_collate(self,batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
        """
        targets = []
        imgs = []
        if self.batch_count % 5 == 0:
            self.rescale_size = random.choice(range(3, 6, 1)) / 10
        #print(self.rescale_size)
        for _, sample in enumerate(batch):
            # print("sample:",sample)
            for _, tup in enumerate(sample):
                if torch.is_tensor(tup):
                    #tup = tup.astype(np.float32)
                    tup=tup.numpy()
                    #print("tup type",type(tup))
                    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                                      cv2.INTER_LANCZOS4]
                    interp_method = interp_methods[random.randrange(5)]
                    #print(tup.shape,self.rescale_size)
                    #tup=cv2.resize(tup,None,fx=self.rescale_size,fy=self.rescale_size,interpolation=interp_method)
                    tup = cv2.resize(tup, (int(my_img_dim*self.rescale_size),int(my_img_dim*self.rescale_size)), interpolation=interp_method)
                    #tup0 = tup
                    #print("tup:",tup.shape)
                    #cv2.imshow('tup',tup)
                    #cv2.waitKey()
                    tup-=(127, 127, 127)
                    tup/= 127
                    tup=tup.transpose(2,0,1)
                    tup=torch.from_numpy(tup)
                    #print("tup:", tup.shape)
                    imgs.append(tup)
                elif isinstance(tup, type(np.empty(0))):
                    # tupa = tup
                    # h, w, _ = tup0.shape
                    # b_w_t = (tupa[:, 2] - tupa[:, 0]) * w
                    # b_h_t = (tupa[:, 3] - tupa[:, 1]) * h
                    # mask_b = np.minimum(b_w_t, b_h_t) > 4.0
                    # tup = tup[mask_b]
                    annos = torch.from_numpy(tup).float()
                    #print("annos:", annos.shape, annos)
                    targets.append(annos)
        self.batch_count += 1
        #print(torch.stack(imgs, 0).shape)
        return (torch.stack(imgs, 0), targets)


class DLDetection(data.Dataset):
    def __init__(self,root,txtname,preproc = None,target_transform = None):
        self.root = root
        self.txtname = txtname
        self.preproc = preproc
        self.rescale_size = 1
        self.target_transform = target_transform
        dst,lst = search(txtname)
        self.imageLst = lst
        self.imageDst = dst
        self.batch_count = 0


    def __len__(self):
        return len(self.imageLst)

    def __getitem__(self, index):
        img_id = random.choice(self.imageLst)

        img = cv2.imread(os.path.join(self.root,img_id),cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        height,width,_ = img.shape
        
        if self.target_transform is not None:
            targets = self.target_transform(self.imageDst[img_id])

        if self.preproc is not None:
            self.preproc.img_dim = (self.rescale_size * my_img_dim)
            img,targets = self.preproc(img,targets)



        return torch.from_numpy(img),targets
    def detection_collate(self,batch):
        targets = []
        imgs = []
        if self.batch_count % 5 == 0:
            self.rescale_size = random.choice(range(3, 6, 1)) / 10
            # self.rescale_size = np.random.uniform(0.3, 0.5)
            # self.rescale_size =0.3
        # print(batch[0][0].shape,batch[1][0].shape)
        for _, sample in enumerate(batch):
            for _, tup in enumerate(sample):
                if torch.is_tensor(tup):
                    # tup = tup.astype(np.float32)
                    tup = tup.numpy()
                    # print("tup type",type(tup))
                    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                                      cv2.INTER_LANCZOS4]
                    interp_method = interp_methods[random.randrange(5)]
                    # print(tup.shape,self.rescale_size)
                    # tup=cv2.resize(tup,None,fx=self.rescale_size,fy=self.rescale_size,interpolation=interp_method)
                    tup = cv2.resize(tup, (int(my_img_dim * self.rescale_size), int(my_img_dim * self.rescale_size)),
                                     interpolation=interp_method)
                    tup0 = tup
                    # print("tup:",tup.shape)
                    # cv2.imshow('tup',tup)
                    # cv2.waitKey()
                    # tup-=(104, 117, 123)
                    tup -= (127,127,127)
                    tup /= 127
                    # tup = np.expand_dims(tup, 2)
                    # print(tup.shape)
                    tup = tup.transpose(2, 0, 1)
                    tup = torch.from_numpy(tup)
                    # print("tup:", tup.shape)
                    imgs.append(tup)
                elif isinstance(tup, type(np.empty(0))):
                    tupa = tup
                    h, w,_ = tup0.shape
                    b_w_t = (tupa[:, 2] - tupa[:, 0]) * w
                    b_h_t = (tupa[:, 3] - tupa[:, 1]) * h
                    mask_b = np.minimum(b_w_t, b_h_t) > 4.0
                    tup = tup[mask_b]
                    annos = torch.from_numpy(tup).float()
                    # print("annos:", annos.shape, annos)
                    targets.append(annos)
        self.batch_count += 1
        # print(torch.stack(imgs, 0).shape)
        return (torch.stack(imgs, 0), targets)



def _resize_subtract_mean(image, insize, rgb_mean = 127):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                print("tup:",tup.shape)
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                print("annos:",annos.shape,annos)
                targets.append(annos)
    return (torch.stack(imgs, 0), targets)
if __name__ == '__main__':
    #dataset = VOCDetection('/home/codingbo/FaceBoxes.PyTorch-master/data/WIDER_FACE', preproc(img_dim, rgb_means), AnnotationTransform())
    img_dim = 1024
    rgb_means = (104, 117, 123)  # bgr order
    dataset_2 = Face4k('/home/codingbo/WORK/data/4K-Face', preproc(img_dim, rgb_means))
    batch_iterator = iter(data.DataLoader(dataset_2, 2, shuffle=True, num_workers=1, collate_fn=detection_collate))
    images, targets = next(batch_iterator)
