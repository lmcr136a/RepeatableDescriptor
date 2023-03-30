import torch


def descriptor_loss_custom(desc, desc_w, kpts, matched_kpts_idx, kpts_w, matched_kpts_idx_w):
    """
    Loss for "one" batch
    desc: (256, H, W)
    kpts: (n, 2)
    matched_kpts_idx: (n'), # of matched points
    """
    n = len(kpts)
    D = desc[:, kpts[matched_kpts_idx.int(), 1], 
                    kpts[matched_kpts_idx.int(), 0]]
    D_w = desc_w[:, kpts_w[matched_kpts_idx_w.int(), 1], 
                        kpts_w[matched_kpts_idx_w.int(), 0]]
    DtD = D.T@D_w
    Ds = torch.diag(torch.diagonal(DtD))
    loss_desc = (torch.sum(DtD)-2*torch.sum(Ds))/n
    return loss_desc