import torch


def get_kpts_from_hm(hm):
    rs, cs = torch.where(hm[0] > 0)
    kpts = torch.stack((cs, rs), dim=1)
    return kpts


def pt_in_list(pt, dblist):
    for xvar in [-1, 0, 1]:
        for yvar in [-1, 0, 1]:
            if [pt[0]+xvar, pt[1]+yvar] in dblist:
                return [pt[0]+xvar, pt[1]+yvar]
    return False


def get_desc_of_kpts(desc, kpts, kpt_idx=None):
    # desc: 256 * H * W
    # kpts: n * 2
    if kpt_idx:
        D = desc[:, kpts[kpt_idx.int(), 1], 
                        kpts[kpt_idx.int(), 0]]
    else:
        D = desc[:, kpts[:, 1], kpts[:, 0]]
    return D


def get_matches(kpts2d, kpts3d, kpts2d_w, kpts3d_w, kpts_output, kpts_output_w, device):
    dbs= kpts2d.to(device)
    dbs_w= kpts2d_w.to(device)
    dbs, dbs_w = (dbs*0.5).int().float(), (dbs_w*0.5).int().float()

    output_kpts_idx = []
    output_3Dcoor = []
    dbslist = dbs.detach().cpu().tolist()
    dbslist_w = dbs_w.detach().cpu().tolist()
    for i, pt in enumerate(kpts_output.detach().cpu().tolist()):
        pt = pt_in_list(pt=pt, dblist=dbslist)
        if pt:
            output_kpts_idx.append(i)            # output_kpts_idx: [2, 3, 5, 6, 7, 9, ...] db에 있는 kpt 인덱스
            output_3Dcoor.append(kpts3d[dbslist.index(pt)])   # [2번의3D좌표, 3번의 3D좌표, 5번의 3D좌표, ...]

    output_kpts_idx_w = []
    output_3Dcoor_w = []
    for i, pt_w in enumerate(kpts_output_w.detach().cpu().tolist()):
        pt_w = pt_in_list(pt=pt_w, dblist=dbslist_w)
        if pt_w:
            output_kpts_idx_w.append(i)
            output_3Dcoor_w.append(kpts3d_w[dbslist_w.index(pt_w)])

    matched_idx, matched_idx_w = [], []
    for i1, k1 in enumerate(output_3Dcoor):
        for i2, k2 in enumerate(output_3Dcoor_w):
            if torch.sum(torch.abs(k1-k2)) < 5.0e-2:
    #             print(k1, k2)
                matched_idx.append(i1)
                matched_idx_w.append(i2)
    matched_kpts_idx = torch.Tensor(output_kpts_idx)[matched_idx]
    matched_kpts_idx_w = torch.Tensor(output_kpts_idx_w)[matched_idx_w]
    return matched_kpts_idx, matched_kpts_idx_w