from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
from opt import opt

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


if __name__ == "__main__":
    facedet = FaceDetection()
    cap = cv2.VideoCapture('test_data/20200522164730261_0.avi')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result_with_mask.avi', fourcc, 20.0, (frame_width, frame_height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            rect = facedet.detection_image(frame)
            for data in rect:
                cv2.rectangle(frame, (data[1], data[2]), (data[3], data[4]), (0, 255, 0), 1)
                cv2.putText(frame, str(data[0]), (data[1], data[2] + 30), font, 1.2, (0, 255, 0), 1)
            out.write(frame)
    cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # cap.release()
    # out.release()
    np.uint8
    print("finish")
    # img = cv2.imread("a.jpg")
    # rect = facedet.detection_image(img)
    # print(rect)



        




        
