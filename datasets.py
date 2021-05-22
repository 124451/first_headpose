import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter,ImageDraw
from tqdm.std import tqdm

import utils
import matplotlib.pyplot as plt

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


class Synhead(Dataset):
    def __init__(self, data_dir, csv_path, transform, test=False):
        column_names = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.X_train = tmp_df['path']
        self.y_train = tmp_df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']]
        self.length = len(tmp_df)
        self.test = test

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.X_train.iloc[index]).strip('.jpg') + '.png'
        img = Image.open(path)
        img = img.convert('RGB')

        x_min, y_min, x_max, y_max, yaw, pitch, roll = self.y_train.iloc[index]
        x_min = float(x_min); x_max = float(x_max)
        y_min = float(y_min); y_max = float(y_max)
        yaw = -float(yaw); pitch = float(pitch); roll = float(roll)

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        width, height = img.size
        # Crop the face
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        return self.length

class RandomErasing(object):
    # 概率，sl下采样便捷，sh上采样边界，图像大小比率
    def __init__(self, p=0.4, sl=0.02, sh=0.3, r1=0.3, r2=2):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):

        if np.random.rand() > self.p:
            return img

        img = np.array(img)

        while True:
            img_h, img_w, img_c = img.shape

            img_area = img_h * img_w
            mask_area = np.random.uniform(self.sl, self.sh) * img_area
            mask_aspect_ratio = np.random.uniform(self.r1, self.r2)
            mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
            mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))

            mask = np.random.rand(mask_h, mask_w, img_c) * 255

            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            right = left + mask_w
            bottom = top + mask_h

            if right <= img_w and bottom <= img_h:
                break

        img[top:bottom, left:right, :] = mask
        # img = img.astype(np.int)
        return Image.fromarray(img)

class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.35 to 0.650
        k = np.random.random_sample() * 0.35 + 0.3
        w,h = x_max - x_min,y_max - y_min
        ratio = h/w - 1
        x_min -= (ratio/2*w+k*h)
        y_min -= (k*h+40)
        # x_min -= 0.6 * k * abs(x_max - x_min)
        # y_min -= 2 * k * abs(y_max - y_min)
        x_max += (ratio/2*w+k*h)
        y_max += (k*h-40)

        # x_min -= 0.6 * k * abs(x_max - x_min)
        # y_min -= 2 * k * abs(y_max - y_min)
        # x_max += 0.6 * k * abs(x_max - x_min)
        # y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)
        # erase
        img = RandomErasing()(img)


        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length


class Pose_300W_LP_random_ds(Dataset):
    # 300W-LP dataset with random downsampling
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        ds = 1 + np.random.randint(0,4) * 5
        original_size = img.size
        img = img.resize((img.size[0] / ds, img.size[1] / ds), resample=Image.NEAREST)
        img = img.resize((original_size[0], original_size[1]), resample=Image.NEAREST)

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length


class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img.astype(np.uint8))
        # img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        ih,iw = img.height,img.width
        x_list = pt2d[0, :]
        x_list = np.sort(x_list)
        r_x_list = x_list[np.argsort(-x_list)]
        for i in range(len(x_list)):
            if x_list[i] != -1:
                x_min = x_list[i]
                break
        for i in range(len(x_list)):
            if r_x_list[i]<iw:
                x_max = r_x_list[i]
                break
        y_list = pt2d[1,:]
        y_list = np.sort(y_list)
        r_y_list = y_list[np.argsort(-y_list)]
        for i in range(len(y_list)):
            if y_list[i] != -1:
                y_min = y_list[i]
                break
        for i in range(len(y_list)):
            if ih>r_y_list[i]:
                y_max = r_y_list[i]
                break
        w,h =  x_max-x_min,y_max-y_min
        ratio = h/w
        k = 0.4
        if ratio >1:
            x_min = max(x_min - w*(ratio-1.0)/2.0 - h*k,0)
            x_max = min(x_max + w*(ratio-1.0)/2.0 + h*k,iw)
            y_min = max(y_min - h*k-35.0,0)
            y_max = min(y_max + h*k-35.0,ih)
        else:
            ratio = w/h
            y_min = max(y_min - h*(ratio-1.0)/2.0 - w*k-35.0,0)
            y_max = min(y_max + h * (ratio - 1.0) / 2.0 + w*k-35.0,ih)
            x_min = max(x_min - w*k,0)
            x_max = min(x_max + w*k,iw)

        # x_min = min(pt2d[0, :])
        # y_min = min(pt2d[1, :])
        # x_max = max(pt2d[0, :])
        # y_max = max(pt2d[1, :])

        # k = 0.20
        # x_min -= 2 * k * abs(x_max - x_min)
        # y_min -= 2 * k * abs(y_max - y_min)
        # x_max += 2 * k * abs(x_max - x_min)
        # y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


class AFLW2000_ds(Dataset):
    # AFLW2000 dataset with fixed downsampling
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        ds = 3  # downsampling factor
        original_size = img.size
        img = img.resize((img.size[0] / ds, img.size[1] / ds), resample=Image.NEAREST)
        img = img.resize((original_size[0], original_size[1]), resample=Image.NEAREST)

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


class AFLW_aug(Dataset):
    # AFLW dataset with flipping
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1

        # Augment
        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length


class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length


class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length


class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.png', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + '_rgb' + self.img_ext))
        img = img.convert(self.image_mode)
        pose_path = os.path.join(self.data_dir, self.y_train[index] + '_pose' + self.annot_ext)

        y_train_list = self.y_train[index].split('/')
        bbox_path = os.path.join(self.data_dir, y_train_list[0] + '/dockerface-' + y_train_list[-1] + '_rgb' + self.annot_ext)

        # Load bounding box
        # bbox = open(bbox_path, 'r')
        with open(bbox_path, 'r') as tf:
            for tdata in tf.readlines():
                line = tdata.split(" ")
                if float(line[1]) > 215.0:
                    break
        

        # line = bbox.readline().split(' ')
        if len(line) < 4:
            x_min, y_min, x_max, y_max = 0, 0, img.size[0], img.size[1]
        else:
            x_min, y_min, x_max, y_max = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        # bbox.close()

        # Load pose in degrees
        pose_annot = open(pose_path, 'r')
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)

        R = np.array(R)
        T = R[3,:]
        R = R[:3,:]
        pose_annot.close()

        R = np.transpose(R)

        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        # Loosely crop face
        k = 0.35
        # k = 0.35
        w,h = x_max - x_min,y_max-y_min
        ratio =  h/w - 1

        x_min -= ((ratio/2.0 * w)+k*h)#w*k*0.6    k*(y_max - y_min)
        y_min -= (k*h+10)
        # y_min -= (k*abs(y_max - y_min)+0)#h*k

        x_max += (ratio/2.0 * w)+k*h#w*h*0.6  + k*(y_max - y_min)
        y_max += (k*h-10)

        # x_min -= 0.6 * k * abs(x_max - x_min)
        # y_min -= k * abs(y_max - y_min)
        # x_max += 0.6 * k * abs(x_max - x_min)
        # y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        # erase
        img = RandomErasing()(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length

#BIWI和300W_LP整合
class BIWI_Pose_300W_LP(Dataset):
    def __init__(self, data_dir, filename_path, transform, biwi_img_ext='.png', biwi_annot_ext='.txt',\
        t300w_lp_img_ext='.jpg', t300w_lp_annot_ext='.mat', image_mode='RGB'):

        self.data_dir = data_dir
        self.transform = transform
        self.biwi_img_ext = biwi_img_ext
        self.biwi_annot_ext = biwi_annot_ext

        self.t300w_lp_img_ext = t300w_lp_img_ext
        self.t300w_lp_annot_ext = t300w_lp_annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        
        file_name,flag= self.X_train[index].split(",")
        ann_name,_ = self.y_train[index].split(",")
        yaw, pitch, roll = 0,0,0
        if int(flag) == 1:
            img = cv2.imread(os.path.join(self.data_dir, file_name + '_rgb' + self.biwi_img_ext))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img.astype(np.uint8))
            # img = Image.open(os.path.join(self.data_dir, file_name + '_rgb' + self.biwi_img_ext))
            img = img.convert(self.image_mode)
            # orial_img = img.copy()

            pose_path = os.path.join(self.data_dir, ann_name + '_pose' + self.biwi_annot_ext)

            y_train_list = ann_name.split('/')
            y_train_new = y_train_list[0:-1]
            temp = ''
            for idt in y_train_new:
                temp = os.path.join(temp,idt)
            y_train_new = temp
            bbox_path = os.path.join(self.data_dir, y_train_new + '/dockerface-' + y_train_list[-1] + '_rgb' + self.biwi_annot_ext)

        # Load bounding box
        # bbox = open(bbox_path, 'r')
            with open(bbox_path, 'r') as tf:
                for tdata in tf.readlines():
                    line = tdata.split(" ")
                    if float(line[1]) > 215.0:
                        break
        

        # line = bbox.readline().split(' ')
            if len(line) < 4:
                x_min, y_min, x_max, y_max = 0, 0, img.size[0], img.size[1]
            else:
                x_min, y_min, x_max, y_max = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        # bbox.close()

        # Load pose in degrees
            pose_annot = open(pose_path, 'r')
            R = []
            for line in pose_annot:
                line = line.strip('\n').split(' ')
                l = []
                if line[0] != '':
                    for nb in line:
                        if nb == '':
                            continue
                        l.append(float(nb))
                    R.append(l)

            R = np.array(R)
            T = R[3,:]
            R = R[:3,:]
            pose_annot.close()

            R = np.transpose(R)

            roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
            yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
            pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

            # Loosely crop face
            k = np.random.random_sample() * 0.3 + 0.15
            # k = 0.35
            w,h = x_max - x_min,y_max-y_min
            ratio =  h/w - 1

            x_min -= ((ratio/2.0 * w)+k*h)#w*k*0.6    k*(y_max - y_min)
            y_min -= (k*h+10)
            # y_min -= (k*abs(y_max - y_min)+0)#h*k

            x_max += (ratio/2.0 * w)+k*h#w*h*0.6  + k*(y_max - y_min)
            y_max += (k*h-10)

            # x_min -= 0.6 * k * abs(x_max - x_min)
            # y_min -= k * abs(y_max - y_min)
            # x_max += 0.6 * k * abs(x_max - x_min)
            # y_max += 0.6 * k * abs(y_max - y_min)
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Bin values
            bins = np.array(range(-99, 102, 3))
            binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

            
            # Flip?
            # rnd = np.random.random_sample()
            # if rnd < 0.5:
            #     yaw = -yaw
            #     roll = -roll
            #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)
            # orial_img = img.copy()
            # erase
            # img = RandomErasing()(img)
            if self.transform is not None:
                img = self.transform(img)
            labels = torch.LongTensor(binned_pose)
            cont_labels = torch.FloatTensor([yaw, pitch, roll])
            # file_name = torch.tensor(file_name)
            # return img, labels, cont_labels, file_name
        else:
            img = cv2.imread(os.path.join(self.data_dir, file_name + self.t300w_lp_img_ext))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img.astype(np.uint8))
            # img = Image.open(os.path.join(self.data_dir, file_name + '_rgb' + self.biwi_img_ext))
            # img = img.convert(self.image_mode)
            # img = Image.open(os.path.join(self.data_dir, file_name + self.t300w_lp_img_ext))
            img = img.convert(self.image_mode)
            
            mat_path = os.path.join(self.data_dir, ann_name + self.t300w_lp_annot_ext)

            # Crop the face loosely
            pt2d = utils.get_pt2d_from_mat(mat_path)
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])

            # k = 0.35 to 0.650
            k = np.random.random_sample() * 0.35 + 0.3
            w,h = x_max - x_min,y_max - y_min
            ratio = h/w - 1
            x_min -= (ratio/2*w+k*h)
            y_min -= (k*h+40)
            # x_min -= 0.6 * k * abs(x_max - x_min)
            # y_min -= 2 * k * abs(y_max - y_min)
            x_max += (ratio/2*w+k*h)
            y_max += (k*h-40)

            # x_min -= 0.6 * k * abs(x_max - x_min)
            # y_min -= 2 * k * abs(y_max - y_min)
            # x_max += 0.6 * k * abs(x_max - x_min)
            # y_max += 0.6 * k * abs(y_max - y_min)
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # We get the pose in radians
            pose = utils.get_ypr_from_mat(mat_path)
            # And convert to degrees.
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi

            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)
            # erase
            # img = RandomErasing()(img)
            # orial_img = img.copy()

            # Bin values
            bins = np.array(range(-99, 102, 3))
            binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

            # Get target tensors
            labels = torch.LongTensor(binned_pose)
            cont_labels = torch.FloatTensor([yaw, pitch, roll])

            if self.transform is not None:
                img = self.transform(img)
        # print(file_name)
        file_name = file_name.split('/')[-1]
        return img, labels, cont_labels, file_name
        # return img, labels, cont_labels, file_name,orial_img,[x_min,y_min,x_max,y_max]

    def __len__(self):
        # 15,667
        return self.length


class myself_dataset(Dataset):
    def __init__(self,filename_path, transform,img_mode = 'RGB'):
        self.transform = transform
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.image_mode = img_mode
    def __getitem__(self,index):
        get_data = self.X_train[index].split(',')
        img_path = get_data[0]
        img = Image.open(img_path)
        img = img.convert(self.image_mode)
        x_min,y_min,x_max,y_max = float(get_data[1]),float(get_data[2]),float(get_data[3]),float(get_data[4])
        yaw,pitch,roll = float(get_data[5]),float(get_data[6]),float(get_data[7])


        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.05:
        #     img = img.filter(ImageFilter.BLUR)
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)
        filename = img_path.split('/')[-1]
        return img, labels, cont_labels,filename
    def __len__(self):
        # 122,450
        return self.length



def main():
    data_dir = "/media/omnisky/D4T/huli/work/headpose/data"
    file_name = "/media/omnisky/D4T/huli/work/headpose/data/file_name_biwi_300w_lp_no_mask20210212.txt"
    data = BIWI_Pose_300W_LP(
                data_dir,
                file_name,
                transform=None
            )
    val_names=""#63340
    for i in tqdm(range(0,len(data))):
        val_imgs,val_labels,val_const_labels,val_names,orial_img,label = data.__getitem__(i)
        # draw_img = ImageDraw.ImageDraw(orial_img)
        # draw_img.rectangle((label[0],label[1],label[2],label[3]),outline='red',width=2)
        img = cv2.cvtColor(np.array(orial_img),cv2.COLOR_RGB2BGR)
        utils.draw_axis(img,val_const_labels[0], val_const_labels[1], val_const_labels[2], tdx = (img.shape[0])//2, tdy= (img.shape[1])//2, size=50)
        str_yan="yan:{:.3f}".format(val_const_labels[0])
        str_pitch="pitch:{:.3f}".format(val_const_labels[1])
        str_roll = "roll:{:.3f}".format(val_const_labels[2])
        cv2.putText(img,str_yan,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
        cv2.putText(img,str_pitch,(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
        cv2.putText(img,str_roll,(0,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
        cv2.imshow("qew",img)
        trs = cv2.waitKey(0)
        if trs==ord('q'):
            break
        # plt.imshow(orial_img)
        # plt.show()
def show_valid():
    val_data_dir = "data/AFLW2000/"
    val_filename_list = "data/AFLW2000/filename_list.txt"
    valid_pose_dataset = AFLW2000(
            val_data_dir,
            val_filename_list,
            None)
    for i in tqdm(range(0,len(valid_pose_dataset))):
        img, labels, cont_labels, fil_name = valid_pose_dataset.__getitem__(i)
        img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        utils.draw_axis(img,cont_labels[0], cont_labels[1], cont_labels[2], tdx = (img.shape[0])//2, tdy= (img.shape[1])//2, size=50)
        str_yan="yan:{:.3f}".format(cont_labels[0])
        str_pitch="pitch:{:.3f}".format(cont_labels[1])
        str_roll = "roll:{:.3f}".format(cont_labels[2])
        cv2.putText(img,str_yan,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
        cv2.putText(img,str_pitch,(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
        cv2.putText(img,str_roll,(0,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
        cv2.imshow("qew",img)
        trs = cv2.waitKey(0)
        if trs==ord('q'):
            break

if __name__ == "__main__":
    show_valid()
        