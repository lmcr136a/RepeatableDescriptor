import torch
import cv2
import random
import torchvision
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os

from PIL import Image

def measure_t(start, before=None, i_now=None, i_total=None, others=None, get_str=False):
    now = time.time()
    runt = now-start
    h = runt//3600
    m = (runt - h*3600)//60
    s = round(runt - h*3600 - m*60)
    if i_now and i_total:
        prtstr = f"Iter {i_now+1} in {i_total}, \
                {round((i_now+1)/i_total, 3) * 100}%"
    else: 
        prtstr = f"Runtime {h}h {m}m {s}s"

    if before:
        runt2 = before-start
        h2 = runt2//3600
        m2 = (runt2 - h2*3600)//60
        s2 = round(runt2 - h2*3600 - m2*60)
        prtstr += f" ({h2}h {m2}m {s2}s)"

    if others:
        prtstr += f"  {others}"
    print(prtstr)
    if get_str:
        return now, prtstr
    return now 

def get_kpts_from_hm(hm, mask=None):
    if str(type(mask)) == "<class 'NoneType'>":
        mask = torch.ones(hm.shape)

    hm = hm.squeeze().detach().cpu()
    mask = mask[0]

    rs, cs = torch.where(hm*mask > 0)
    kpts = torch.stack((cs, rs), dim=1).tolist()
    return kpts


def apply_random_H(im1):
    theta = 30                        # rotation, -30~30 degree
    t = 100                           # translation, -100~100
    p = 100                            # perspective, -10~10
    theta=(np.random.rand(1)*2*theta-theta)[0] # -30~30
    t_x, t_y = (np.random.rand(2)*2*t-t)[:]    # -100~100
    scale= (np.random.rand(1)*0.8 + 0.6)[0] # 0.7~1.5
    
    H, W = im1.shape[1:]
    startpoints=[[0,0],[0,W], [H, 0], [H, W]]
    startpoints=np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    startpoints *= np.array([H,W])
    
    endpoints = []
    for pt in startpoints:
        xvar, yvar = np.random.rand(2)*2*p-p
        endpoints.append([pt[0]+xvar, pt[1]+yvar])
    im1 = torchvision.transforms.functional.perspective(im1, 
                                startpoints=startpoints,  
                                endpoints=endpoints,
                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    
    dft_a, dft_t, dft_s, dft_sh = 0, [0,0], 1, [0,0]
    
    im1 = torchvision.transforms.functional.affine(im1, angle=theta, translate=dft_t,      scale=dft_s, shear=dft_sh)
    im1 = torchvision.transforms.functional.affine(im1, angle=dft_a, translate=[t_x, t_y], scale=dft_s, shear=dft_sh)
    im1 = torchvision.transforms.functional.affine(im1, angle=dft_a, translate=dft_t,      scale=scale, shear=dft_sh)
    
    Hinv_info = {
        'theta':-theta, 't_x':-t_x, 't_y':-t_y, 'scale':1/scale,
        'startpoints':endpoints, 'endpoints':startpoints
            }
    
    return im1, Hinv_info


def apply_H_from_info(im1, Hinfo): ## for inverse H, 이미지는 배치 가능인데 Hinfo는 배치로 안됨
    dft_a, dft_t, dft_s, dft_sh = 0, [0,0], 1, [0,0]
    im1 = torchvision.transforms.functional.affine(im1, angle=dft_a, translate=dft_t,      scale=Hinfo['scale'], shear=dft_sh)
    im1 = torchvision.transforms.functional.affine(im1, angle=dft_a, translate=[Hinfo['t_x'], Hinfo['t_y']], scale=dft_s, shear=dft_sh)
    im1 = torchvision.transforms.functional.affine(im1, angle=Hinfo['theta'], translate=dft_t,      scale=dft_s, shear=dft_sh)

    im1 = torchvision.transforms.functional.perspective(im1, 
                                startpoints=Hinfo['startpoints'],  
                                endpoints=Hinfo['endpoints'],
                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    return im1


def apply_random_H_batch(im1):
    theta = 30                        # rotation, -30~30 degree
    t = 100                           # translation, -100~100
    p = 100                            # perspective, -10~10
    theta=(np.random.rand(1)*2*theta-theta)[0] # -30~30
    t_x, t_y = (np.random.rand(2)*2*t-t)[:]    # -100~100
    scale= (np.random.rand(1)*0.8 + 0.6)[0] # 0.7~1.5
    
    H, W = im1.shape[2:]
    
    startpoints=[[0,0],[0,W], [H, 0], [H, W]]
    startpoints=np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    startpoints *= np.array([H,W])

    
    endpoints = []
    for pt in startpoints:
        xvar, yvar = np.random.rand(2)*2*p-p
        endpoints.append([pt[0]+xvar, pt[1]+yvar])
        
    im1 = torchvision.transforms.functional.perspective(im1, 
                                startpoints=startpoints,  
                                endpoints=endpoints,
                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    
    dft_a, dft_t, dft_s, dft_sh = 0, [0,0], 1, [0,0]
    
    im1 = torchvision.transforms.functional.affine(im1, angle=theta, translate=dft_t,      scale=dft_s, shear=dft_sh)
    im1 = torchvision.transforms.functional.affine(im1, angle=dft_a, translate=[t_x, t_y], scale=dft_s, shear=dft_sh)
    im1 = torchvision.transforms.functional.affine(im1, angle=dft_a, translate=dft_t,      scale=scale, shear=dft_sh)
    
    Hinv_info = {
        'theta':-theta, 't_x':-t_x, 't_y':-t_y, 'scale':1/scale,
        'startpoints':endpoints, 'endpoints':startpoints
            }
    
    return im1, Hinv_info


def sort_by_similar_pts(kpts1, kpts2, desc1, desc2):
    # sort kpts2 correspondent to kpts1
    ary = torch.cdist(desc2, desc1)
    mask1 = ary == torch.unsqueeze(torch.min(ary,axis=1).values, dim=1)
    mask2 = ary == torch.min(ary, axis=0).values
    idx1, idx2 = torch.where(mask1*mask2)
    color = ary[idx1, idx2]
    color = 255-color*200
    color = torch.clamp(color.int(), 0, 255)
    color = torch.stack([torch.zeros_like(color), color, color], axis=1)
    return kpts1[idx2], kpts2[idx1], color.detach().cpu().tolist()


def pt_in_list(pt, dblist):
    for xvar in [-1, 0, 1]:
        for yvar in [-1, 0, 1]:
            if [pt[0]+xvar, pt[1]+yvar] in dblist:
                return [pt[0]+xvar, pt[1]+yvar]
    return False


def get_matches(kpts2d, kpts3d, kpts2d_w, kpts3d_w, kpts_output, 
                kpts_output_w, device, thd=1.5):
    # kpts2d:       N * 2
    # kpts3d:       N * 3
    # kpts_output:  homo_batch * N_pred * 2 ,  list type!
    # if len(np.array(kpts_output).shape) == 2:
    #     kpts_output = [kpts_output]
    #     kpts_output_w = [kpts_output_w]
    # homo_batch = len(kpts_output)
    dbslist, dbslist_w = np.array(kpts2d), np.array(kpts2d_w)
    kpts_output, kpts_output_w = np.array(kpts_output), np.array(kpts_output_w)

    # matched_kpts_idx, matched_kpts_idx_w = [], []
    # for homo_iter in range(homo_batch):
    output_kpts_idx = []
    output_3Dcoor = []
    # for i, pt in enumerate(kpts_output[homo_iter]):
    for i, pt in enumerate(kpts_output):
        # pt = pt_in_list(pt=pt, dblist=dbslist)\
        dis = np.sqrt(np.sum(np.power(dbslist-pt, 2), axis=1))
        v = np.min(dis)
        idx = np.argmin(dis)
        if v < 2:
            output_kpts_idx.append(i)            # output_kpts_idx: [2, 3, 5, 6, 7, 9, ...] db에 있는 kpt output 인덱스
            output_3Dcoor.append(kpts3d[idx])   # [2번의3D좌표, 3번의 3D좌표, 5번의 3D좌표, ...]

    output_kpts_idx_w = []
    output_3Dcoor_w = []
    # for i, pt_w in enumerate(kpts_output_w[homo_iter]):
    for i, pt_w in enumerate(kpts_output_w):
        dis = np.sqrt(np.sum(np.power(dbslist_w-pt_w, 2), axis=1))
        v = np.min(dis)
        idx = np.argmin(dis)
        if v < 2:
            output_kpts_idx_w.append(i)            # output_kpts_idx: [2, 3, 5, 6, 7, 9, ...] db에 있는 kpt output 인덱스
            output_3Dcoor_w.append(kpts3d_w[idx])   # [2번의3D좌표, 3번의 3D좌표, 5번의 3D좌표, ...]

    matched_idx, matched_idx_w = [], []
    output_3Dcoor_w = np.array(output_3Dcoor_w)
    output_3Dcoor = np.array(output_3Dcoor)

    torch.save(output_3Dcoor, 'output_3Dcoor.pt')
    torch.save(output_3Dcoor_w, 'output_3Dcoor_w.pt')
    torch.save(output_kpts_idx_w, 'output_kpts_idx_w.pt')
    torch.save(output_kpts_idx, 'output_kpts_idx.pt')

    torch.save(kpts_output_w, 'kpts_output_w.pt')
    torch.save(kpts_output, 'kpts_output.pt')
    torch.save(kpts2d, 'kpts2d.pt')
    torch.save(kpts3d, 'kpts3d.pt')
    torch.save(kpts2d_w, 'kpts2d_w.pt')
    torch.save(kpts3d_w, 'kpts3d_w.pt')
    
    print(output_3Dcoor_w.shape)
    print(np.array(output_3Dcoor).shape)
    for i1, k1 in enumerate(output_3Dcoor):
        dis = np.sqrt(np.sum(np.power(output_3Dcoor_w - k1, 2), axis=1))
        v = np.min(dis)
        idx = np.argmin(dis)
        print(v)
        if v < 2:
            matched_idx.append(i1)
            matched_idx_w.append(idx)
    # matched_kpts_idx.append(torch.Tensor(output_kpts_idx)[matched_idx].tolist())
    # matched_kpts_idx_w.append(torch.Tensor(output_kpts_idx_w)[matched_idx_w].tolist())
    matched_kpts_idx = torch.Tensor(output_kpts_idx)[matched_idx].tolist()
    matched_kpts_idx_w = torch.Tensor(output_kpts_idx_w)[matched_idx_w].tolist()
    return matched_kpts_idx, matched_kpts_idx_w


def get_dup_kpts(kpts, kpts3d, cnt_thd=1):
    # at least twice dup pts of kpts2d and corresponding 3dkpts
    unq, idx, cnt = np.unique(kpts, axis=0, return_inverse=True, return_counts=True)
    cnt_mask = cnt > cnt_thd
    dup_kpts = unq[cnt_mask]
    # indices for kpts3d
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(idx, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx = np.argsort(idx[idx_mask])
    dup_idx = np.split(idx_idx[srt_idx], np.cumsum(cnt[cnt_mask])[:-1])
    dup_idx = [d[0] for d in dup_idx]
    dup_kpts3d = kpts3d[dup_idx]
    return dup_kpts, dup_kpts3d

def get_homo_img_cat(img, HOMOGRAPHY_NUM):
    im_trf_cat, Hinv_infos = torch.Tensor(), []
    for homography_iter in range(HOMOGRAPHY_NUM):
        im_trf, Hinv_info = apply_random_H_batch(img)
        Hinv_infos.append(Hinv_info)
        if im_trf_cat.shape[0] == 0:
            im_trf = torch.unsqueeze(im_trf, dim=1)
            im_trf_cat = im_trf
        else:
            im_trf = torch.unsqueeze(im_trf, dim=1)
            im_trf_cat = torch.cat([im_trf_cat, im_trf], axis=1)
    return im_trf_cat, Hinv_infos


def get_Hidx(i, HOMO_BATCH, HOMO_NUM):
    Hidx1, Hidx2 = i, i+HOMO_BATCH
    if Hidx2 > HOMO_NUM:
        Hidx2 = HOMO_NUM
    return Hidx1, Hidx2

