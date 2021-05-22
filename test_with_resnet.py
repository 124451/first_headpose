from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils_face.nms_wrapper import nms
import cv2
from models.faceboxes import FaceBoxes
from utils_face.box_utils import decode
from utils_face.timer import Timer
from opt import opt
import time
from model_resnet import ResidualNet
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
import utils
class FaceDetection(object):
    def __init__(self):
        self.cfg = cfg
        self.model = opt["model"]
        self.use_cpu = opt["use_cpu"]
        self.mean = opt["mean"]
        self.val = opt["val"]
        self.confidence_threshold = opt["confidence_threshold"]
        self.nms_threshold = opt["nms_threshold"]
        self.top_k = opt["top_k"]
        self.keep_top_k = opt["keep_top_k"]
        self.yuzhi = opt["yuzhi"]
        self.device = torch.device("cpu" if self.use_cpu else "cuda")
        self.weights = FaceBoxes(phase="test",size = None,num_classes = 2)
        self.net = self.load_model(self.weights,self.model,self.use_cpu)

    def load_model(self,weights,model,use_cpu):
        # print("load model from {}".format(model))
        if use_cpu:
            pretrained_dict = torch.load(model, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(model, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(weights, pretrained_dict)
        weights.load_state_dict(pretrained_dict, strict=False)

        weights.eval()
        
        weights = weights.to(self.device)
        return weights

    def remove_prefix(self,state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def check_keys(self,model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print('Missing keys:{}'.format(len(missing_keys)))
        # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        # print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def search_jpg(self,directory):
        dst  = {}
        for curdir,subdir,files in os.walk(directory):
            for jpeg in (file for file in files if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg")):
                path = os.path.join(curdir,jpeg)
                dst[jpeg] = path
        return dst
    
    def detection_image(self,image):
        """
        in:mat data
        out:lst([[score,xmin,ymin,xmax,ymax]])

        """
        lst = []
        
        if len(image.shape) == 3:
            h,w,_ = image.shape
        elif len(image.shape) == 2:
            h,w = image.shape
        else:
            return 0
        image_resize = cv2.resize(image,(256,int(256*h/w)))
        image_resize = np.float32(image_resize)
        if len(image_resize.shape) == 3:
            im_height,im_width,_ = image_resize.shape
        elif len(image_resize.shape) == 2:
            im_height,im_width = image_resize.shape
        
        scale = torch.Tensor([w,h,w,h])

        image_resize -= self.mean
        image_resize /= self.val

        img = image_resize.transpose(2,0,1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        out = self.net(img)
        priorbox = PriorBox(self.cfg,out[2],(im_height,im_width),phase="test")
        priors = priorbox.forward()
        priors = priors.to(self.device)
        loc,conf,_ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.data.cpu().numpy()[:,1]
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes1 = boxes[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes1 = boxes1[order]
        scores = scores[order]
        dets = np.hstack((boxes1, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_threshold,force_cpu=self.use_cpu)
        dets = dets[keep, :]
        dets = dets[:self.keep_top_k,:]       
        for k in range(dets.shape[0]):
            face_rectangle = {}
            xmin = dets[k,0]
            ymin = dets[k,1]
            xmax = dets[k,2]
            ymax = dets[k,3]
            score = dets[k,4]
            if score > self.yuzhi:
                lst.append([score,int(xmin),int(ymin),int(xmax),int(ymax)])

        return lst

def detect_head_pose224():
    facedet = FaceDetection()
    #model path
    snapshot_path = "output/no_mask_02_gray_mix_biwi_300w_lp_cosin_224/gray_biwi_300W_LP_squire_epoch_24.pkl"
    cap = cv2.VideoCapture('test_data/20200522164730261_0.avi')

    model = ResidualNet("ImageNet", 50, 66, "CBAM")
    new_state_dict = OrderedDict()
    saved_state_dict = torch.load(snapshot_path)
    for k, v in saved_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda(0)
    model.eval()
    transformations = transforms.Compose(
        [transforms.Scale(224),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.392,0.392,0.392],
             std=[0.254, 0.254, 0.254])])



    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result_data/result_with_mask.avi', fourcc, 20.0, (frame_width, frame_height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            rect = facedet.detection_image(frame)
            h,w = 0,0
            k = 0.35
            b = 30
            if len(rect)==0:
                out.write(frame)
                continue
            for i, data in enumerate(rect):
                temp_w,temp_h = data[3]-data[1],data[4]-data[2]
                if (i==0) or (h*w<temp_h*temp_w):
                    h,w = temp_h,temp_w
                    x1,y1,x2,y2 = data[1],data[2],data[3],data[4]
                ratio = h/w
                if ratio > 1:
                    ratio = ratio - 1
                    x1 -= (ratio/2*w+k*h)
                    y1 -= (k*h+b)
                    x2 += (ratio/2*w+k*h)
                    y2 += (k*h-b)
                    # crop_img.append(img.crop((int(x1),int(ymin),int(x1),int(ymax))))
                else:
                    ratio = w/h - 1
                    x1 -= (k*w)
                    y1 -= (ratio/2*h+k*w+b)
                    x2 += (k*w)
                    y2 += (ratio/2*h + k*w-b)
            crop_img = frame[int(y1):int(y2)+1,int(x1):int(x2)+1]
            # change to rgb
            cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB,crop_img)
            detect_img = Image.fromarray(crop_img)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                # cv2.putText(frame, str(data[0]), (data[1], data[2] + 30), font, 1.2, (0, 255, 0), 1)
            
            # head pose
            idx_tensor = [idx for idx in range(66)]
            idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)
            yaw_error = .0
            pitch_error = .0
            roll_error = .0
            with torch.no_grad():
                detect_img = transformations(detect_img)
                detect_img = detect_img.unsqueeze(dim=0)
                yaw, pitch, roll = model(detect_img.cuda(0))
            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
            utils.draw_axis(frame,yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx = (x2-x1)//2+x1, tdy= (y2-y1)//2+y1, size=50)
            put_ptich_str = "ptich:{:.4f}".format(pitch_predicted[0])
            put_yaw_str = "yaw:{:.4f}".format(yaw_predicted[0])
            cv2.putText(frame, str(put_ptich_str), (int(x1), int(y1)-30), font, 1, (0, 255, 0), 1)
            cv2.putText(frame, str(put_yaw_str), (int(x1), int(y1)-60), font, 1, (0, 255, 0), 1)
            out.write(frame)
        else:
            print("finish")
            out.release()
            break
    
    # cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # cap.release()
    # out.release()
    # np.uint8
    print("finish")
    # img = cv2.imread("a.jpg")
    # rect = facedet.detection_image(img)
    # print(rect)
def detect_headpose_resnet_112():
    facedet = FaceDetection()
    #model path
    snapshot_path = "output/no_mask_03_gray_biwi_300w_lp_cosin_112/gray_biwi_300W_LP_squire_epoch_30.pkl"
    cap = cv2.VideoCapture(' test_data/20200522164730261_0.avi')

    model = ResidualNet("ImageNet", 50, 66, "CBAM")
    new_state_dict = OrderedDict()
    saved_state_dict = torch.load(snapshot_path)
    for k, v in saved_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda(0)
    model.eval()
    transformations = transforms.Compose(
        [transforms.Scale(112),
        #  transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.392,0.392,0.392],
             std=[0.254, 0.254, 0.254])])



    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result_data/result_with_mask_resnet_112.avi', fourcc, 20.0, (frame_width, frame_height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            rect = facedet.detection_image(frame)
            h,w = 0,0
            k = 0.35
            b = 30
            if len(rect)==0:
                out.write(frame)
                continue
            for i, data in enumerate(rect):
                temp_w,temp_h = data[3]-data[1],data[4]-data[2]
                if (i==0) or (h*w<temp_h*temp_w):
                    h,w = temp_h,temp_w
                    x1,y1,x2,y2 = data[1],data[2],data[3],data[4]
                ratio = h/w
                if ratio > 1:
                    ratio = ratio - 1
                    x1 -= (ratio/2*w+k*h)
                    y1 -= (k*h+b)
                    x2 += (ratio/2*w+k*h)
                    y2 += (k*h-b)
                    # crop_img.append(img.crop((int(x1),int(ymin),int(x1),int(ymax))))
                else:
                    ratio = w/h - 1
                    x1 -= (k*w)
                    y1 -= (ratio/2*h+k*w+b)
                    x2 += (k*w)
                    y2 += (ratio/2*h + k*w-b)
            crop_img = frame[int(y1):int(y2)+1,int(x1):int(x2)+1]
            # change to rgb
            cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB,crop_img)
            detect_img = Image.fromarray(crop_img)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                # cv2.putText(frame, str(data[0]), (data[1], data[2] + 30), font, 1.2, (0, 255, 0), 1)
            
            # head pose
            idx_tensor = [idx for idx in range(66)]
            idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)
            yaw_error = .0
            pitch_error = .0
            roll_error = .0
            with torch.no_grad():
                detect_img = transformations(detect_img)
                detect_img = detect_img.unsqueeze(dim=0)
                yaw, pitch, roll = model(detect_img.cuda(0))
            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
            utils.draw_axis(frame,yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx = (x2-x1)//2+x1, tdy= (y2-y1)//2+y1, size=50)
            put_ptich_str = "ptich:{:.4f}".format(pitch_predicted[0])
            put_yaw_str = "yaw:{:.4f}".format(yaw_predicted[0])
            cv2.putText(frame, str(put_ptich_str), (int(x1), int(y1)-30), font, 1, (0, 255, 0), 1)
            cv2.putText(frame, str(put_yaw_str), (int(x1), int(y1)-60), font, 1, (0, 255, 0), 1)
            out.write(frame)
        else:
            print("finish")
            out.release()
            break
    
    # cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # cap.release()
    # out.release()
    # np.uint8
    print("finish")
    # img = cv2.imread("a.jpg")
    # rect = facedet.detection_image(img)
    # print(rect)



if __name__ == "__main__":
    # detect_headpose_resnet_112()
    detect_head_pose224()


        




        
