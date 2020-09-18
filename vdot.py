import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils import data
from matplotlib.image import imread
class VdotDataset(data.Dataset):
    def __init__(self, image_dir, ann_file):

        
        self.image_dir=image_dir
        self.ann_file = ann_file
        #ann = open('/home/ekta/AI_current/vdot/vdot/train_annotations/train_annotations.json', 'r')
        ann = open(self.ann_file)
        data_json = json.load(ann)
        self.data_json = data_json
        #image_dir = os.listdir('/home/ekta/AI_current/vdot/vdot/train_set')
        image_dir=os.listdir(self.image_dir)
        total_num_images = len(image_dir)
        self.total_num_images = total_num_images
        self.imgs_list, self.annot_list = self.parse_labels(self.data_json)
        #self.imgs_list = []
        #self.annot_list= []
        
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

            det_dict = {'bbox' : np.concatenate([v]), 'classes' : np.array([1])}
            frame_boxes.append(det_dict)    
        #all_boxes.append(frame_boxes)

        "write the logic so as to not to repeat the 70 annotations"
        "considering the drop_inlets and storm_drains as a single asset item" 
        return filename_labels, frame_boxes

    
    def __len__(self):
       return self.total_num_images


    def __getitem__(self, index):
        self.image_name = self.imgs_list[index]
        #img= Image.open(os.path.join('/home/ekta/AI_current/vdot/vdot/revised_set', self.image_name))
        labels = self.annot_list[index]
        #labels = np.array([labels])
        #img = imread(os.path.join('/home/ekta/AI_current/vdot/vdot/train_set', self.image_name))
        img = imread(os.path.join(self.image_dir, self.image_name))
        width = 512
        height = 512
        dim = (width, height)
        img = cv2.resize(img, dim) #interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= img.transpose(2,0,1)
        self.sample= {'img' : img, 'labels' : labels}
        
        return img, labels





    














