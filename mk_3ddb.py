import yaml
import os
import time
import logging
import cv2
import torchvision

import torch
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import open3d as o3d
from tensorboardX import SummaryWriter

from utils.utils import getWriterPath, get_log_dir
from settings import EXPER_PATH

## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *
from copy import deepcopy as dc
from utils.d2s import DepthToSpace, SpaceToDepth
from utils.utils import flattenDetection
from Train_model_frontend_cubemap import thd_img

from train_cubemap import *
from utils_custom import *
from utils_custom_visualize import *

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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

seed = 1325
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

fnc = 1023.5
args = Namespace(command='train_joint', config='configs/magicpoint_cubemap.yaml', debug=False, eval=False, exper_name='cubemap_dataset', func=train_joint)


with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
output_dir = get_log_dir('logs')
############################################### train_joint

torch.set_default_tensor_type(torch.FloatTensor)
task = config['data']['dataset']
config['model']['batch_size'] = 1
config['model']['eval_batch_size'] = 1
config['data']['not_warped_images'] = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info('train on device: %s', device)
with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
# writer = SummaryWriter(getWriterPath(task=args.command, date=True))
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

# feed the data into the agent
train_agent.train_loader = train_loader
train_agent.val_loader = val_loader

# load model initiates the model and load the pretrained model (if any)
train_agent.loadModel()
train_agent.dataParallel()

camera_matrix = torch.Tensor([[fnc,0,fnc],[0,fnc,fnc],[0,0,1]])

net = dc(train_agent.net)

def get_blurred(img, thd=0.015):
    blurrer = torchvision.transforms.GaussianBlur(9, 2)
    blurred = blurrer(img)
    edge=img[0]-blurred
    edge_1 = torch.where(edge>thd, 1.0, 0)
    edge_2 = torch.where(edge<-thd, 1.0, 0)
    blurred = (edge_1+edge_2)/2
    blurred = torch.mean(blurred, axis=(0))
    blurred = blurrer(torch.unsqueeze(blurred, dim=0))
    return blurred[0]


def get_line_kpts(kpt, kpt3d, img):
    torch.save(img, 'img.pt')
    blurred = get_blurred(img)  # img: [3, 1024, 1024]
    newkpt, newkpt3d = [],[]
    torch.save(blurred, 'blurred.pt')
    for i, pt in enumerate(kpt):
        v = blurred[pt[0], pt[1]]
        if v>0:
            newkpt.append(list(pt))
            newkpt3d.append(list(kpt3d[i]))
    return newkpt, newkpt3d

def visualize_kpts(imname1, imname2, kpts1, kpts2, name='_', n=3000):
    if type(imname1) == str:
        im1 = np.array(Image.open(imname1).resize((1024,1024)))
    else:
        im1 = imname1.transpose(0, 1).transpose(1,2).detach().cpu().numpy()
    if type(imname2) == str:
        im2 = np.array(Image.open(imname2).resize((1024,1024)))
    else:
        im2 = imname2.transpose(0, 1).transpose(1,2).detach().cpu().numpy()
    R = 2
    fig, ax = plt.subplots(1, 2) 
    
    ax[0].imshow(im1)
    i = 0

    for x, y in kpts1:
        ax[0].add_patch(plt.Circle((x, y), R, color='r'))
        # ax[0].text(x, y, str(i))
        i += 1
        if i>=n:
            break
            
    i=0
    ax[1].imshow(im2)
    for x, y in kpts2:
        ax[1].add_patch(plt.Circle((x, y), R, color='r'))
        # ax[1].text(x, y, str(i))
        i += 1
        if i>=n:
            break
    plt.title(f'{len(kpts1)}kpts, {len(kpts2)}kpts.')
    plt.savefig(os.path.join('figures', f'{name}.png'))

RTfile = {}

HOMO_NUM, HOMO_BATCH = 100, 20
thd = 0.2

DBNAME = f'KeyPts2D3D_1024_H_thd{thd}_cnt2_230411.mat'
# DBNAME2 = f'KeyPts2D3D_1024_H_thd{thd}_cnt3.mat'
processed_img_names = []
io.savemat(DBNAME, RTfile)
# io.savemat(DBNAME2, RTfile)
for _iter, sample in enumerate(train_loader):
    img = sample['image'].to(device)
    batch_size = img.shape[0]
    kpts2d_total = []
    kpts3d_total = []

    b=0
    ptcloud_list = []
    # for b in range(batch_size):   # usually batch_size = 1
    target = o3d.io.read_point_cloud(sample['ply_path'][b])
    points = torch.Tensor(np.array(target.points)).to(device)
    del target
    ptcloud_list.append(points)

    with torch.inference_mode():
        for h in range(0, HOMO_NUM, HOMO_BATCH):
            Hidx1, Hidx2 = get_Hidx(h, HOMO_BATCH, HOMO_NUM)
            H_NUM_THIS = Hidx2-Hidx1

            im_trf_cat, Hinv_infos = get_homo_img_cat(img, H_NUM_THIS)

            # for b in range(batch_size): # batch size
            if sample['img_path'][b] not in processed_img_names:
                out = net(im_trf_cat[b])
                hms = flattenDetection(out['semi'])
                for hb in range(H_NUM_THIS):
                    hm = apply_H_from_info(hms[hb], Hinv_infos[hb])
                    hm = thd_img(hm, thd=thd)
                    kpts = get_kpts_from_hm(hm)
                    coor3D = take3Dpoint(ptcloud_list[b], torch.Tensor(kpts).to(device), sample['R'][b].to(device).float(), 
                                sample['T'][b].to(device), camera_matrix, device)
                    kpts2d_total.append(kpts)
                    kpts3d_total.append(coor3D)

        # for b in range(batch_size):
        ipath = sample['img_path'][b]

        # for i in range(0, len(kpts2d_total), 1):
        #     visualize_kpts(ipath, im_trf_cat[b][i], torch.Tensor(kpts2d_total[i]), [],
        #                    name=f"iter{_iter}_{i}{i+1}")
        #     print(i, 'saved')

        kpts2d_total = [onept for onehomo in kpts2d_total for onept in onehomo ]
        kpts3d_total = [onept for onehomo in kpts3d_total for onept in onehomo ]

        RTfile = io.loadmat(DBNAME)
        if ipath not in list(RTfile.keys()):
            kpttotal, kpttotal3d = get_dup_kpts(kpts2d_total, 
                                                np.array(kpts3d_total),
                                                cnt_thd=2)
            # visualize_kpts(ipath, ipath, torch.Tensor(kpttotal), torch.Tensor(kpttotal),
            #                name=f"iter{_iter}_kpttotal2")

            # kpttotal, kpttotal3d = get_line_kpts(kpttotal, kpttotal3d, img[b])
            RTfile.update({ipath+'_3Dkpts': kpttotal3d})
            RTfile.update({ipath+'2Dkpts': kpttotal})

        io.savemat(DBNAME, RTfile)
        # RTfile = io.loadmat(DBNAME2)
        # if ipath not in list(RTfile.keys()):
        #     kpttotal, kpttotal3d = get_dup_kpts(kpts2d_total[b], 
        #                                         np.array(kpts3d_total[b]),
        #                                         cnt_thd=3)
        #     RTfile.update({ipath+'_3Dkpts': kpttotal3d})
        #     RTfile.update({ipath+'2Dkpts': kpttotal})
        
        # io.savemat(DBNAME2, RTfile)
        processed_img_names.append(sample['img_path'][b])
        print(f"{' '*30} {len(kpttotal)} ,, {len(kpttotal3d)}")

    runt = time.time()-start
    h = runt//3600
    m = (runt - h*3600)//60
    s = round(runt - h*3600 - m*60)
    print(f"iter {_iter+1} in {len(train_loader)}, \
        {round((_iter+1)/len(train_loader), 4) * 100}%\
            {h}h {m}m {s}s\
            ")

print("FINISH")
exit()

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
