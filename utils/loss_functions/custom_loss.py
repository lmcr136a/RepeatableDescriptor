import torch
import torch.nn as nn
from utils_custom import *

def get_desc_of_kpts(desc, kpts, kpt_idx=[]):
    # desc: 256 * H * W
    # kpts: n * 2
    kpts = torch.Tensor(kpts).int()
    if kpt_idx==[]:
        D = desc[:, kpts[:, 1], 
                    kpts[:, 0]]
    else:
        D = desc[:, kpts[kpt_idx.int(), 1], 
                    kpts[kpt_idx.int(), 0]]
    return D


def descriptor_loss_custom(desc, desc_w, kpts, matched_kpts_idx, kpts_w, matched_kpts_idx_w):
    """
    Loss for "one" batch
    desc: (hb, 256, H, W)
    kpts: (hb, n, 2)
    matched_kpts_idx: (hb, n'), # of matched points
    """
    kpts, kpts_w, matched_kpts_idx, matched_kpts_idx_w = torch.Tensor(kpts), torch.Tensor(kpts_w), torch.Tensor(matched_kpts_idx), torch.Tensor(matched_kpts_idx_w)
    n = len(kpts)
    D = get_desc_of_kpts(desc, kpts, matched_kpts_idx)
    D_w = get_desc_of_kpts(desc_w, kpts_w, matched_kpts_idx_w)

    DtD = D.T@D_w
    Ds = torch.diag(torch.diagonal(DtD))
    loss_desc = (torch.sum(DtD)-2*torch.sum(Ds))/n
    return loss_desc

# def descriptor_loss_custom_batch(desc, desc_w, kpts, matched_kpts_idx, kpts_w, matched_kpts_idx_w, device):
#     l = torch.Tensor([0])[0].to(device)
#     for hb in range(desc.shape[0]):
#         l += descriptor_loss_custom(desc[hb], desc_w[hb], kpts[hb], 
#                                     matched_kpts_idx[hb], kpts_w[hb], matched_kpts_idx_w[hb])
    
#     return l


def detection_loss_custom(semi, labels3D_in_loss, mask_3D_flattened, device="cpu"):
    semi = torch.unsqueeze(semi, dim=0)
    labels3D_in_loss = torch.unsqueeze(labels3D_in_loss, dim=0) ## torch crossentropyloss 는 다차원인경우 배치 포함한 경우만 취급
    loss_func = nn.CrossEntropyLoss(reduce=False).to(device)
    loss = loss_func(semi, labels3D_in_loss)
    loss = (loss * mask_3D_flattened).sum()
    loss = loss / (mask_3D_flattened.sum() + 1e-10)
    return loss


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


def labels2Dto3D(labels, cell_size, add_dustbin=True): 
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    labels = torch.unsqueeze(labels, dim=1)
    B, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels)
    if add_dustbin:
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.] = 0
        labels = torch.cat((labels, dustbin.view(B, 1, Hc, Wc)), dim=1)
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
    return labels


def labels2Dto3D_flattened(labels, cell_size=8):  ## batch
    '''
    Change the shape of labels into 3D. Batch of labels.
    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, c,  H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(cell_size)
    labels = space2depth(labels)
    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    labels = torch.cat((labels*2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.argmax(labels, dim=1)
    return labels


def compute_valid_mask(image_shape, Hinv_info, device='cpu', erosion_radius=2):
    mask = torch.ones(*image_shape).to(device)
    mask = apply_H_from_info(mask, Hinv_info)
    mask = mask.view(*image_shape)
    mask = mask.cpu().numpy()
    # if erosion_radius > 0:
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
    #     for i in range(image_shape[0]):
    #         mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)
    return torch.tensor(mask).to(device)


def getMasks(imgshape, Hinv_info, edgepx_rm=20, cell_size=8, device="cpu"):
    """
    # 2D mask is constructed into 3D (Hc, Wc) space for training
    :param imgshape:
        (H, W)
    :return:
        flattened 3D mask for training
    """
    b,h,w = imgshape
    Hinv_info['endpoints'] = rmedge(Hinv_info['endpoints'], edgepx_rm)
    mask2D = compute_valid_mask((1,h,w), Hinv_info, device=device)  # 1은 채널을 뜻함.. 배치가 아니지만 1배치로 코드가 작성되어있음
    mask_3D = labels2Dto3D(
        mask2D.to(device), cell_size=cell_size, add_dustbin=False
    ).float()
    mask_3D_flattened = torch.prod(mask_3D, 1)  #  [1, 128, 128]
    return mask_3D_flattened, mask2D.detach().cpu()


def getLabels(img_shape, labels_1D, cell_size=8, device="cpu"):
    """
    labels_2D: N * 2, N: kpt num
    """
    labels_2D = torch.zeros(img_shape[0], 1, img_shape[1], img_shape[2]).to(device)
    labels_2D[:, :,  labels_1D[:,0].int(), labels_1D[:,1].int()] = 1
    labels3D_flattened = labels2Dto3D_flattened(
        labels_2D.to(device), cell_size=cell_size
    )
    return labels3D_flattened

def rmedge(p4list, px):
    new = []
    for p in p4list:
        e1, e2 = -px, -px
        if p[0] == 0:
            e1 *= -1
        if p[1] == 0:
            e2 *= -1
        new.append([p[0]+e1, p[1]+e2])
    return new
