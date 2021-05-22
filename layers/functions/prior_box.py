import torch
from itertools import product as product
import numpy as np


class PriorBox(object):
    def __init__(self, cfg, box_dimension=None, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.variance = cfg['variance']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

        self.feature_maps = box_dimension.cpu().numpy().astype(np.int)
        self.image_size = image_size
        '''
        if phase == 'train':
            self.image_size = (cfg['min_dim'], cfg['min_dim'])
            self.feature_maps = cfg['feature_maps']
        elif phase == 'test':
            self.feature_maps = box_dimension.cpu().numpy().astype(np.int)
            self.image_size = image_size
        '''
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        #print(self.feature_maps)
        for k, f in enumerate(self.feature_maps):
            #print(f[0],f[1])
            if k==3:
               break
            min_sizes = self.min_sizes[k]
            
            for i, j in product(range(f[0]), range(f[1])):   #f[1] j 对应 x 也就是宽度
                for min_size in min_sizes:

                    #s_kx = min_size / self.image_size[1]
                    #s_ky = min_size / self.image_size[0]
                    center_x=(j+0.5)*self.steps[k]
                    center_y=(i+0.5)*self.steps[k]
                    #print("center_x:",center_x,",center_y:",center_y)
                    if min_size == 32:
                        '''
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            mean += [cx, cy, s_kx, s_ky]
                        '''
                        for m in range(-2,2,1):
                            for n in range(-2,2,1):
                                minx=(center_x+n*8-(min_size-1)/2)/self.image_size[1]
                                miny = (center_y + m * 8 - (min_size - 1) / 2) / self.image_size[0]
                                maxx = (center_x + n * 8 + (min_size - 1) / 2) / self.image_size[1]
                                maxy = (center_y + m * 8 + (min_size - 1) / 2) / self.image_size[0]
                                mean+=[minx,miny,maxx,maxy]
                    elif min_size == 64:
                        '''
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            mean += [cx, cy, s_kx, s_ky]
                        '''
                        for m in range(-1,1,1):
                            for n in range(-1,1,1):
                                minx=(center_x+n*16-(min_size-1)/2)/self.image_size[1]
                                miny = (center_y + m * 16 - (min_size - 1) / 2) / self.image_size[0]
                                maxx = (center_x + n * 16 + (min_size - 1) / 2) / self.image_size[1]
                                maxy = (center_y + m * 16 + (min_size - 1) / 2) / self.image_size[0]
                                mean+=[minx,miny,maxx,maxy]
                    else:
                        '''
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        mean += [cx, cy, s_kx, s_ky]
                        '''
                        minx = (center_x  - (min_size - 1) / 2) / self.image_size[1]
                        miny = (center_y  - (min_size - 1) / 2) / self.image_size[0]
                        maxx = (center_x  + (min_size - 1) / 2) / self.image_size[1]
                        maxy = (center_y  + (min_size - 1) / 2) / self.image_size[0]
                        mean += [minx, miny, maxx, maxy]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
