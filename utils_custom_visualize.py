import torch
import cv2
import random
import torchvision
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import time
import os

from PIL import Image


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
        if img.shape[0] in [3, 1]: # rgb or grayscale
            img = np.transpose(img, (1,2,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        

def showkpts(im1, kpts1, n=200):
    R = 2
    ax = plt.subplot()
    ax.imshow(im1)
    i=0
    for x, y in kpts1:
        ax.add_patch(plt.Circle((x, y), R, color='r'))
        ax.text(x, y, str(i))
        i+=1
        if i> n:
            break
    plt.show()


def visualize(imname1, imname2, kpts1=[], kpts2=[], n=3, name='sample'):
    if type(imname1) == str:
        im1, im2 = np.array(Image.open(imname1).resize((1024,1024))), np.array(Image.open(imname2).resize((1024,1024)))
    else:
        im1, im2 = np.array(imname1), np.array(imname2)

    if im1.shape[0] in [1, 3]:
        im1 = np.transpose(im1, (1,2,0))
        im2 = np.transpose(im2, (1,2,0))
    R = 2
    fig, ax = plt.subplots(1, 2) 
    
    ax[0].imshow(im1)
    i = 0
    for x, y in kpts1:
        ax[0].add_patch(plt.Circle((x, y), R, color='r'))
#         ax[0].text(x, y, str(i))
        i += 1
        if i>=n:
            break
            
    i=0
    ax[1].imshow(im2)
    for x, y in kpts2:
        ax[1].add_patch(plt.Circle((x, y), R, color='r'))
#         ax[1].text(x, y, str(i))
        i += 1
        if i>=n:
            break
    plt.title(f"{len(kpts1)}kpts, {len(kpts2)}kpts")
    plt.savefig(f'{name}.png')


def show_sample_from_db(dbname='KeyPts2D3D_1024_H_thd0.2_cnt2_edge.mat', idxs=None):
    RTfile1 = io.loadmat(dbname)
    li = list(RTfile1.keys())
    li.sort()
    print("DB length:", len(li))
    if not idxs:
        idx1 = random.sample(li[3::2], 1)[0]
        idx2 = random.sample(li[3::2], 1)[0]
    else:
        idx1, idx2 = idxs
    imname1 = idx1.split('2D')[0]
    kpts1 = RTfile1[idx1]
    imname2 = idx2.split('2D')[0]
    kpts2 = RTfile1[idx2]

    visualize(imname1, imname2, kpts1, kpts2, n=10000)   


def maching_plot(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color=None, path=None,
                            show_keypoints=True, margin=10,
                            img_shape=(1024,1024)):
    if type(image0) == str:
        image0, image1 = Image.open(image0).convert("L"), Image.open(image1).convert("L")
        image0, image1 = np.array(image0.resize(img_shape)), np.array(image1.resize(img_shape))
    else:
        if image0.shape[1] == 3:
            image0, image1 = image0[0], image1[0]
        image0, image1 = torch.mean(image0, dim=0), torch.mean(image1, dim=0)
        image0, image1 = image0.detach().cpu().numpy(), image1.detach().cpu().numpy()
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0*255
    out[:H1, W0+margin:] = image1*255
    out = np.stack([out]*3, -1)
    
    R1, R2, th = 4,3,1
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0.int().detach().cpu().numpy()), np.round(kpts1.int().detach().cpu().numpy())
        white = (255, 0, 0)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), R1, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), R2, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), R1, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), R2, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0.int().detach().cpu().numpy()), np.round(mkpts1.int().detach().cpu().numpy())
    
    if color is None:
        color = [(255, 255, 0)]* len(mkpts0)

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=th, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), R2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    # visualize(image0, image1, kpts0, kpts1, n=10000)   

    return out