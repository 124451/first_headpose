import cv2
import numpy as np
import random
from utils_face.box_utils import matrix_iof
# import imgaug.augmenters as iaa

def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        #若gt的中心落在crop后的区域，则该gt保留
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        '''
        # ignore tiny faces
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 16.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue
        '''
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # ignore tiny faces  忽略在输入图下长宽小于16的gt
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 8.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue
        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag

def _crop1(image):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):

        if random.uniform(0, 1) <= 0.7:
            scale = 1
        else:
            scale = random.uniform(0.7, 1.)
        #print(scale)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))


        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]



        return image_t
    return image


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes
def _mirror1(image):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]

    return image

def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _pad_to_square1(image, rgb_mean):

    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)



def gasuss_noise(image, mean=0, var=0.001):
  image = np.array(image/255, dtype=float)
  noise = np.random.normal(mean, var ** 0.5, image.shape)
  out = image + noise
  # if out.min() < 0:
  #   low_clip = -1.
  # else:
  low_clip = 0.
  out = np.clip(out, low_clip, 1.0)
  out = np.uint8(out*255)
  #cv.imshow("gasuss", out)
  return out

def gamma_trans(img, gamma):

    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)

# def mode_illumination(img, num):
#     choicenum = random.choice(range(0,num))
#     if choicenum == 0:
#         aratio = int(np.random.uniform(-40, 20))
#         bratio = np.random.uniform(0.4, 1.6)
#         cratio = np.random.uniform(0.4, 1.5)
#         jratio = np.random.randint(60,80)
#         seq = iaa.Sequential([
#             #iaa.AverageBlur(k=3),
#             #iaa.AddToHueAndSaturation(value=aratio),
#             iaa.JpegCompression(compression=jratio)
#             #iaa.GammaContrast(gamma=cratio),  # crop images from each side by 0 to 16px (randomly chosen)
#             #iaa.LogContrast(gain=bratio, per_channel=True)  # horizontally flip 50% of the images

#             # blur images with a sigma of 0 to 3.0
#             # iaa.SigmoidContrast(gain=(5,15),cutoff=(0.1,1.0))
#         ])
#         # cv2.imshow('pre.jpg',img)
#         img = img[np.newaxis, :]

#         img = seq.augment_images(img)
#         #img = seq(image=img)
#         img = np.squeeze(img)
#         # cv2.imshow('end.jpg',img)
#         # cv2.waitKey()
#     return img

class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        #image = image.astype(np.float32)
        assert targets.shape[0] > 0, "this image does not have gt"
        #print(targets.shape)
        boxes = targets[:, :-1].copy()
        boxes[:,2:] += boxes[:,:2]
        labels = targets[:, -1].copy()

        #image_t = _distort(image)
        #image_t, boxes_t = _expand(image_t, boxes, self.cfg['rgb_mean'], self.cfg['max_expand_ratio'])
        #image_t, boxes_t, labels_t = _crop(image_t, boxes, labels, self.img_dim, self.rgb_means)
        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        image_t = mode_illumination(image_t, 10)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t = _mirror(image_t, boxes_t)
        # cv2.imshow('end.jpg',image_t)
        # cv2.waitKey()
        height, width, _ = image_t.shape

        #image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        image_t = image_t.astype(np.float32)

        #print(boxes_t)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t

class preproc1(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image):


        image_t  = _crop1(image)
        image_t = _distort(image_t)
        image_t = _pad_to_square1(image_t,self.rgb_means)
        image_t = _mirror1(image_t)

        image_t = image_t.astype(np.float32)

        return image_t
