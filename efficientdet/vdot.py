import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import torch.utils.data as data
from torch.utils import data
from matplotlib.image import imread
from pycocotools.coco import COCO
from effdet.data.parsers import create_parser
class VdotDataset(data.Dataset):
    def __init__(self, image_dir,  ann_file, parser=None, parser_kwargs=None, transform=None):
        super(VdotDataset, self).__init__
        parser_kwargs=parser_kwargs or {}
        self.transform = transform
        self.image_dir=image_dir
        self.ann_file = ann_file
        self.coco = None
        #ann = open('/home/ekta/AI_current/vdot/vdot/train_annotations/train_annotations.json', 'r')
        ann = open(self.ann_file)
        data_json = json.load(ann)
        self.data_json = data_json
        #image_dir = os.listdir('/home/ekta/AI_current/vdot/vdot/train_set')
        image_dir=os.listdir(self.image_dir)
        total_num_images = len(image_dir)
        self.total_num_images = total_num_images
        self.imgs_list, self.annot_list = self.parse_labels(self.data_json)
        #self.for_size = for_size
       # self.for_size = self.img_info(self.image_dir)
        #self.imgs_list = []
        #self.annot_list= []
        self._parser = parser
        self._transform = transform
        self.cat_dicts = [{'id': 1, 'name': 'inlet'}]
    '''def parse_images(self, image_dir, ann_file):

        for item in data_json.values():
            if item['filename'] in image_dir:
                img = Image.open(os.path.join('/home/ekta/AI_current/vdot/vdot/train_set', item['filename']))
                for i in item['assets']:
                    if i == 'storm_drain':
                        annot_list.append(i.values() & i['drop_inlet'])
                    else:
                        annot_list.append(i['storm_drain'])
        return img, annot_list '''

    def parse_labels(self, ann_file):

        #annot_lists=[]
        filename_labels = []
        frame_boxes =[]
        det_dict = {}
        all_boxes = []
        
        for i,v in self.data_json.items():
            filename_labels.append(i)
            #annot_lists.append(v)
            v=np.array(v, dtype=np.float32)
            det_dict = {'bbox' : np.concatenate([v]), 'cls' : np.ones(len(v), dtype=np.int64), 'img_size': (800, 600)}
            frame_boxes.append(det_dict)    
        #all_boxes.append(frame_boxes)

        return filename_labels, frame_boxes

    
    def __len__(self):
       return self.total_num_images
    
    def __getitem__(self, index):
        self.image_name = self.imgs_list[index]
        labels = self.annot_list[index]
        labels['img_id'] = int(index)
        #labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        #img = imread(os.path.join('/home/ekta/AI_current/vdot/vdot/train_set', self.image_name))
        img= Image.open(os.path.join(self.image_dir, self.image_name)).convert('RGB')
        '''img = imread(os.path.join(self.image_dir, self.image_name))
        width = 512
        height = 512
        dim = (width, height)
        img = cv2.resize(img, dim) #interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= img.transpose(2,0,1)
        #img /= 255.0
        self.sample= {'img' : img, 'labels' : labels}'''
        if self.transform is not None:
            img, labels = self.transform(img, labels)
        
        return img, labels

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t



    '''def img_info(self, image_dir):
        #os.listdir(self.image_dir)[1]
        img =Image.open(os.listdir(self.image_dir)[1]).convert('RGB')
        imge = img.size
        return imge'''
        

# self.img_info(self.image_dir)

#np.array([800, 600])
    




#'img_size' : img.size 
#'img_scale' : 









