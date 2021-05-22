import os
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
from torchvision import transforms
import utils
from tqdm import tqdm
from datasets import BIWI_Pose_300W_LP

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

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
        wdata = open("/media/omnisky/D4T/huli/work/headpose/data/filename_mix_biwi_300e_lp.txt",'a')


        img_path = os.path.join(self.data_dir, self.X_train[index] + '_rgb' + self.img_ext)
        # img = Image.open(os.path.join(self.data_dir, self.X_train[index] + '_rgb' + self.img_ext))
        # img = img.convert(self.image_mode)
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
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Bin values
        # bins = np.array(range(-99, 102, 3))
        # binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # labels = torch.LongTensor(binned_pose)
        # cont_labels = torch.FloatTensor([yaw, pitch, roll])
        # erase
        # img = RandomErasing()(img)
        if self.transform is not None:
            img = self.transform(img)
        wdata.write("{},{},{},{},{},{},{},{}\n".format(img_path,x_min,y_min,x_max,y_max,yaw,pitch,roll))
        wdata.close()
        return 0
        # return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length
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
        wdata = open("/media/omnisky/D4T/huli/work/headpose/data/filename_mix_biwi_300e_lp.txt",'a')


        img_path = os.path.join(self.data_dir, self.X_train[index] + self.img_ext)
        # img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        # img = img.convert(self.image_mode)
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
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.05:
        #     img = img.filter(ImageFilter.BLUR)
        # erase
        # img = RandomErasing()(img)


        # Bin values
        # bins = np.array(range(-99, 102, 3))
        # binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        # labels = binned_pose
        # cont_labels = torch.FloatTensor([yaw, pitch, roll])

        # if self.transform is not None:
        #     img = self.transform(img)
        wdata.write("{},{},{},{},{},{},{},{}\n".format(img_path,x_min,y_min,x_max,y_max,yaw,pitch,roll))
        wdata.close()
        return 0

    def __len__(self):
        # 122,450
        return self.length

def main():
    # __init__(self, data_dir, filename_path, transform, img_ext='.png', annot_ext='.txt', image_mode='RGB')
    #BIWI
    # data_dir = '/media/omnisky/D4T/huli/work/headpose/data/mask_biwi/'
    # filenme_dir = '/media/omnisky/D4T/huli/work/headpose/data/mask_biwi/img_name.txt'
    # loader = BIWI(data_dir,filenme_dir,None)
    # for count in tqdm(range(len(loader))):
    #     loader.__getitem__(count)
    #Pose_300W_LP
    # data_dir = '/media/omnisky/D4T/huli/work/headpose/data/300W_LP/'
    # filename_dir = '/media/omnisky/D4T/huli/work/headpose/data/300W_LP/filename_list.txt'
    # loader = Pose_300W_LP(data_dir,filename_dir,None)
    # for count in tqdm(range(len(loader))):
    #     loader.__getitem__(count)
    transformations = transforms.Compose(
        [transforms.ColorJitter(brightness=0.85,contrast=0.5,saturation=0.5,hue=0.05)
         ])
    data_dir = '/media/omnisky/D4T/huli/work/headpose/data/'
    filename_dir = '/media/omnisky/D4T/huli/work/headpose/data/Filename_biwi_300w_lp.txt'
    load_dir = BIWI_Pose_300W_LP(data_dir,filename_dir,None)
    means = [0,0,0]
    stdevs = [0,0,0]
    for data in tqdm(range(len(load_dir))):
        
        #将图像保存
        images, labels, cont_labels, name = load_dir.__getitem__(data)
        img = np.array(images).astype(np.float32)/255.0
        for i in range(3):
            means[i] += img[:,:,i].mean()
            stdevs[i] += img[:,:,i].std()
    means = np.asarray(means)/len(load_dir)
    stdevs = np.asarray(stdevs)/len(load_dir)
    print('means={}'.format(means))
    print('stdevs={}'.format(stdevs))
        # if '.' in name:
        #     images.save('./temp_result/'+name)
        # else:
        #     images.save('./temp_result/'+name+'.png')
        
    # batch_iterator = iter(DataLoader(dataset=load_dir,batch_size=100,shuffle=False,num_workers=8))
    
    # for i in range(1535):
    #     print("count:{}".format(i))
    #     images, labels, cont_labels,name = next(batch_iterator)



if __name__ == "__main__":
    main()