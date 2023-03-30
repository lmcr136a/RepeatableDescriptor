import yaml
import os
import time
import logging
import cv2

import torch
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import open3d as o3d
from tensorboardX import SummaryWriter

from utils.utils import getWriterPath
from settings import EXPER_PATH

## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *
from copy import deepcopy as dc
from utils.d2s import DepthToSpace, SpaceToDepth
from utils.utils import flattenDetection
from Train_model_frontend_cubemap import thd_img

from train_cubemap import *


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def take3Dpoint(target_ply, kpts2D, rot, trans, camera_matrix, device):
    '''
    target_ply is numpy array of point cloud (only 3d points!)
    kpts2D is numpy array of 2d keypoints
    rot is rotation matrix of camera (3 by 3)
    trans is translation vector of camera (1 by 3)
    camera_matrix is intrinsic camera matrix (3 by 3)
    '''
    
    ##  
    transformed_target = torch.matmul(rot.T, (target_ply-trans).T).T
    mask = transformed_target[:,0] > 0
    transformed_target_positiveX = transformed_target[mask]
    
    ## 
    kpts2D_h = torch.hstack((kpts2D, torch.ones(len(kpts2D)).reshape(-1,1).to(device)))
    uni_h = torch.matmul(torch.linalg.inv(camera_matrix).to(device), kpts2D_h.T)
    reflection = torch.Tensor([[0,0,1],[-1,0,0],[0,-1,0]]).to(device) # camera coordinates and world coordinates are different!
    uni_w = torch.matmul(reflection, uni_h).T
    
    final3Dpoint = []
    
    # print(f'Get {len(uni_w)} keypoints')
    for i, uni in enumerate(uni_w):
        scale = torch.matmul(transformed_target_positiveX , uni) / (torch.linalg.norm(uni) *torch.linalg.norm(uni))
        # compute displacement vectors from the line
        delta = transformed_target_positiveX - (uni * scale.reshape(-1, 1))[:,:3]
        # compute distances from each vectors to the line
        error = torch.linalg.norm(delta, axis = 1)
        # find the closest point closed to the line
        min_error, argmin_error = torch.min(error), torch.argmin(error)
        # 3D coordinates in camera coordinates
        pt = torch.matmul(rot, transformed_target_positiveX[argmin_error])+trans
        final3Dpoint.append(pt.tolist()[0])
#         print(i, pt, min_error)
    # print('Complete')
    return final3Dpoint



def showim(imgs):
    plt.figure()
    if not type(imgs) == list:
        imgs = [imgs]
    for img in imgs:
        try:
            img = img.detach().cpu().numpy()
        except:
            img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    
start = time.time()
    
fnc = 1023.5
thd = 0.45
args = Namespace(command='train_joint', config='configs/magicpoint_cubemap.yaml', debug=False, eval=False, exper_name='cubemap_dataset', func=train_joint)


with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(EXPER_PATH, args.exper_name)
############################################### train_joint

torch.set_default_tensor_type(torch.FloatTensor)
task = config['data']['dataset']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info('train on device: %s', device)
with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
# writer = SummaryWriter(getWriterPath(task=args.command, date=True))
writer = SummaryWriter(getWriterPath(task=args.command, 
    exper_name=args.exper_name, date=True))
## save data
save_path = get_save_path(output_dir)


data = dataLoader(config, dataset=task, warp_input=True)
train_loader, val_loader = data['train_loader'], data['val_loader']

datasize(train_loader, config, tag='train')
datasize(val_loader, config, tag='val')
# init the training agent using config file
# from train_model_frontend import Train_model_frontend
from utils.loader import get_module
train_model_frontend = get_module('', config['front_end_model'])
train_agent = train_model_frontend(config, save_path=save_path, device=device)

# writer from tensorboard
train_agent.writer = writer

# feed the data into the agent
train_agent.train_loader = train_loader
train_agent.val_loader = val_loader

# load model initiates the model and load the pretrained model (if any)
train_agent.loadModel()
train_agent.dataParallel()

camera_matrix = torch.Tensor([[fnc,0,fnc],[0,fnc,fnc],[0,0,1]])

net = dc(train_agent.net)

# RTfile = {}

DBNAME = 'KeyPts2D3D.mat'

# io.savemat(DBNAME, RTfile)
# for _iter, sample in enumerate(train_loader):

#     img, img_w = sample['image'].to(device), sample['warped_image'].to(device)
#     with torch.no_grad():
#         out = net(img)
#         out_w = net(img_w)
#         hms = thd_img(flattenDetection(out['semi']), thd=thd)
#         hms_w = thd_img(flattenDetection(out_w['semi']), thd=thd)
        
#         for b in range(img.shape[0]): # batch size
#             hm, hm_w = hms[b][0], hms_w[b][0]
#             rs, cs = torch.where(hm > 0)
#             rs_w, cs_w = torch.where(hm_w > 0)
#             kpts = torch.stack((cs, rs), dim=1)
#             kpts_w = torch.stack((cs_w, rs_w), dim=1)

#             target = o3d.io.read_point_cloud(sample['ply_path'][b])

#             points = torch.Tensor(np.array(target.points)).to(device)
#             del target
#             coor3D = take3Dpoint(points, kpts, sample['R'][b].to(device).float(), 
#                         sample['T'][b].to(device), camera_matrix, device)
#             coor3D_w = take3Dpoint(points, kpts_w, sample['R_w'][b].to(device).float(), 
#                         sample['T_w'][b].to(device), camera_matrix, device)
            

        #     RTfile = io.loadmat(DBNAME)
        #     ipath, ipath_w = sample['img_path'][b], sample['img_path_w'][b]

        #     if ipath not in list(RTfile.keys()):
        #         RTfile.update({ipath+'_3Dkpts': coor3D})
        #         RTfile.update({ipath+'2Dkpts': kpts.detach().cpu().tolist()})

        #     if ipath_w not in list(RTfile.keys()):
        #         RTfile.update({ipath_w+'_3Dkpts': coor3D_w})
        #         RTfile.update({ipath_w+'2Dkpts': kpts_w.detach().cpu().tolist()})
        #     # print(RTfile)
        #     io.savemat(DBNAME, RTfile)
        # runt = time.time()-start
        # h = runt//3600
        # m = (runt - h*3600)//60
        # s = round(runt - h*3600 - m*60)
        # print(f"iter {_iter+1} in {len(train_loader)}, \
        #       {round((_iter+1)/len(train_loader), 4) * 100}%\
        #         {h}h {m}m {s}s\
        #         ")




for _iter, sample in enumerate(val_loader):

    net = dc(train_agent.net)
    img, img_w = sample['image'].to(device), sample['warped_image'].to(device)
    with torch.no_grad():

        out = net(img)
        out_w = net(img_w)
        hms = thd_img(flattenDetection(out['semi']), thd=thd)
        hms_w = thd_img(flattenDetection(out_w['semi']), thd=thd)
        
        for b in range(img.shape[0]): # batch size
            hm, hm_w = hms[b][0], hms_w[b][0]
            rs, cs = torch.where(hm > 0)
            rs_w, cs_w = torch.where(hm_w > 0)
            kpts = torch.stack((cs, rs), dim=1)
            kpts_w = torch.stack((cs_w, rs_w), dim=1)

            target = o3d.io.read_point_cloud(sample['ply_path'][b])

            points = torch.Tensor(np.array(target.points)).to(device)
            del target
            coor3D = take3Dpoint(points, kpts, sample['R'][b].to(device).float(), 
                        sample['T'][b].to(device), camera_matrix, device)
            coor3D_w = take3Dpoint(points, kpts_w, sample['R_w'][b].to(device).float(), 
                        sample['T_w'][b].to(device), camera_matrix, device)
            
            RTfile = io.loadmat(DBNAME)
            if _iter < 3:
                print(len(RTfile.keys()))
            ipath, ipath_w = sample['img_path'][b], sample['img_path_w'][b]

            if ipath not in list(RTfile.keys()):
                RTfile.update({ipath+'_3Dkpts': coor3D})
                RTfile.update({ipath+'2Dkpts': kpts.detach().cpu().tolist()})

            if ipath_w not in list(RTfile.keys()):
                RTfile.update({ipath_w+'_3Dkpts': coor3D_w})
                RTfile.update({ipath_w+'2Dkpts': kpts_w.detach().cpu().tolist()})
            # print(RTfile)
            io.savemat(DBNAME, RTfile)
        runt = time.time()-start
        h = runt//3600
        m = (runt - h*3600)//60
        s = round(runt - h*3600 - m*60)
        print(f"iter {_iter+1} in {len(train_loader)}, \
              {round((_iter+1)/len(train_loader), 4) * 100}%\
                {h}h {m}m {s}s\
                ")
