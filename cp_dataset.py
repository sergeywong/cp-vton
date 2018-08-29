#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.mode = opt.mode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.mode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = Image.open(osp.join(self.data_path, 'cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
     
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array > 200).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0)

        # person image 
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
        shape = torch.from_numpy(2*parse_shape-1) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        single_map = Image.new('L', (self.fine_width, self.fine_height))
        single_draw = ImageDraw.Draw(single_map)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.ellipse((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                single_draw.ellipse((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        single_map = self.transform(single_map)
        
        # cloth-agnostic representation
        agnostic = torch.cat([shape.unsqueeze_(0), im_h, pose_map], 0) 

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'pose_img': single_map, # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)



if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--mode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    
    opt = parser.parse_args()
    gmm = CPDataset(opt)

    print('Size of the dataset: ', len(gmm))
    first_item = gmm.__getitem__(0)

    from IPython import embed; embed()

