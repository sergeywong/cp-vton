#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset
from networks import GMM, UnetGenerator

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--checkpoint", default = "")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--mode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument("--display_count", type=int, default = 10)
    parser.add_argument("--keep_epoch", type=int, default = 20)
    parser.add_argument("--decay_epoch", type=int, default = 20)

    
    opt = parser.parse_args()
    return opt

def train_gmm(opt, train_loader, model, board):
    criterionL1 = nn.L1Loss()
    
    for epoch in range(opt.keep_epoch + opt.decay_epoch):
        # optimizer, not well implemented for lr schedular
        if epoch < opt.keep_epoch:
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        else:
            new_lr = opt.lr * (1 - (epoch - opt.keep_epoch) / opt.decay_epoch)
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        for i, inputs in enumerate(train_loader):
            iter_start_time = time.time()
            
            im = inputs['image']
            im_pose = inputs['pose_image']
            im_h = inputs['head']
            shape = inputs['shape']

            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_c =  inputs['parse_cloth'].cuda()
            im_g = inputs['grid_image'].cuda()
            
            grid, theta = model(agnostic, c)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

            loss = criterionL1(warped_cloth, im_c)
            visuals = [ [im_h, shape, im_pose], 
                    [c, warped_cloth, im_c], 
                    [warped_grid, warped_mask,im ] ]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_count = epoch * len(train_loader) + i
            if global_count % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, global_count)
                board.add_scalar('metric', loss.item(), global_count)
                t = time.time() - iter_start_time
                print('epoch: %6d, step: %8d, time: %.3f, loss: %4f' % (epoch, i, t, loss.item()))


def main():
    opt = get_opt()
    print(opt)
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    if False:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.workers, pin_memory=True, sampler=train_sampler)

    # create model
    if opt.stage:
        model = GMM(opt)
    else:
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
   
    model.cuda()
    
    # criterions
    criterionL1 = nn.L1Loss()
    criterionTV = lambda x: (
            torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).sum() + \
            torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).sum() )

    # optimizer
    adam = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # train
    train_gmm(opt, train_loader, model, board)

if __name__ == "__main__":
    print("Start to train geometric matching module!")
    main()
