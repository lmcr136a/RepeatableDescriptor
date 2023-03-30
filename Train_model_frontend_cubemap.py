"""This is the frontend interface for training
base class: inherited by other Train_model_*.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""
import os
import logging
import torch
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.tools import dict_update
from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint, get_log_dir


def thd_img(img, thd=0.015):
    """
    thresholding the image.
    :param img:
    :param thd:
    :return:
    """
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


class Train_model_frontend_cubemap(object):
    """
    # This is the base class for training classes. Wrap pytorch net to help training process.
    
    """

    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        """
        ## default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]
        
        :param config:
            dense_loss, sparse_loss (default)
        """
        # config
        print("Load Train_model_frontend!!")
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False
        self.loss = 0
        self.train_only_descriptor = config["model"].get('train_only_descriptor', False)
        print('----\n----\ntrain_only_descriptor: ', self.train_only_descriptor)
        self.max_iter = config["train_iter"]

        if self.config["model"]["sparse_loss"]["enable"]:
            ## our sparse loss has similar performace, more efficient
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            # from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse
            # self.descriptor_loss = batch_descriptor_loss_sparse

            from utils.loss_functions.custom_loss import descriptor_loss_custom
            self.descriptor_loss = descriptor_loss_custom

            self.desc_loss_type = "sparse"

        self.printImportantConfig()
        self.figname = 0
        self.visualize = False
        pass

    def printImportantConfig(self):
        self.write_log("=" * 10 + " check!!! "+ "=" * 10)
        self.write_log("learning_rate: "+ str(self.config["model"]["learning_rate"]))
        self.write_log("detection_threshold: "+ str(self.config["model"]["detection_threshold"]))
        self.write_log("batch_size: "+ str(self.config["model"]["batch_size"]))
        self.write_log("=" * 10+ " descriptor: "+ str(self.desc_loss_type)+ "=" * 10)

        for item in list(self.desc_params):
            self.write_log(f'{item}, ": ", {self.desc_params[item]}')

        self.write_log("=" * 32)
        pass

    def dataParallel(self):
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = self.net
        self.optimizer = self.adamOptim(
            self.net, lr=self.config["model"]["learning_rate"]
        )
        pass

    def adamOptim(self, net, lr):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        """
        print("adam optimizer")
        import torch.optim as optim

        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer

    def loadModel(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        model = self.config["model"]["name"]
        params = self.config["model"]["params"]
        print("model: ", model)
        net = modelLoader(model=model, **params).to(self.device)
        logging.info("=> setting adam solver")
        optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])

        n_iter = 0
        ## new model or load pretrained
        if self.config["retrain"] == True:
            print("New model")
            pass
        else:
            path = self.config["pretrained"]
            mode = "" if path[-4:] == ".pth" else "full" # the suffix is '.pth' or 'tar.gz'
            print("load pretrained model from: %s", path)
            net, optimizer, n_iter = pretrainedLoader(
                net, optimizer, n_iter, path, mode=mode, full_path=True
            )
            print("successfully load pretrained model from: %s", path)

        def setIter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.net = net
        if self.train_only_descriptor:
            for child in self.net.children():
                for param in child.parameters():
                    param.requires_grad = False
            self.net.convDa.weight.requires_grad = True
            self.net.bnDa.weight.requires_grad = True
            self.net.convDb.weight.requires_grad = True
            self.net.bnDb.weight.requires_grad = True
            print("\n\n Train only descriptor")

        self.optimizer = optimizer
        self.n_iter = setIter(n_iter)
        pass

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    def train(self, **options):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max iterations
        :param options:
        :return:
        """
        # training info
        logging.info("n_iter: %d", self.n_iter)
        logging.info("max_iter: %d", self.max_iter)
        epoch = 0
        # Train one epoch
        while self.n_iter < self.max_iter:
            print("epoch: ", epoch)
            epoch += 1
            self.visualize = True
            #####################3
            print_interval = 100
            loss_batch = []  # To compute average loss of certain batch, batch == print interval

            for i, sample_train in tqdm(enumerate(self.train_loader)):
                loss_out = self.train_val_sample(sample_train, self.n_iter, True)
                self.n_iter += 1
                loss_batch.append(loss_out)
                
                if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                    logging.info("====== Validating...")
                    for j, sample_val in enumerate(self.val_loader):
                        self.train_val_sample(sample_val, self.n_iter + j, False)
                        if j > self.config.get("validation_size", 3):
                            break

                if self.n_iter % print_interval == 0:
                    iterlog = f"iter {self.n_iter}, loss: {np.mean(loss_batch)}"
                    loss_batch = []
                    print("\n", iterlog)
                    self.write_log(iterlog)

                if self.n_iter % self.config["save_interval"] == 0:
                    self.saveModel()

                if self.n_iter > self.max_iter:
                    logging.info("End training: %d", self.n_iter)
                    break

        pass
    def write_log(self, string):
        logfile = os.path.join(self.save_path, 'log.txt')
        with open(logfile, 'a') as f:
            f.write(string)
            f.write("\n")

    def getLabels(self, labels_2D, cell_size, device="cpu"):
        """
        # transform 2D labels to 3D shape for training
        :param labels_2D:
        :param cell_size:
        :param device:
        :return:
        """
        labels3D_flattened = labels2Dto3D_flattened(
            labels_2D.to(device), cell_size=cell_size
        )
        labels3D_in_loss = labels3D_flattened
        return labels3D_in_loss

    def getMasks(self, mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(
            mask_2D.to(device), cell_size=cell_size, add_dustbin=False
        ).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    def get_loss(self, semi, labels3D_in_loss, mask_3D_flattened, device="cpu"):
        """
        ## deprecated: loss function
        :param semi:
        :param labels3D_in_loss:
        :param mask_3D_flattened:
        :param device:
        :return:
        """
        loss_func = nn.CrossEntropyLoss(reduce=False).to(device)
        loss = loss_func(semi, labels3D_in_loss)
        loss = (loss * mask_3D_flattened).sum()
        loss = loss / (mask_3D_flattened.sum() + 1e-10)
        return loss

    def train_val_sample(self, sample, n_iter=0, train=False):
        img, img_w = sample['image'].to(self.device), sample['warped_image'].to(self.device)
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        self.optimizer.zero_grad()
        if train:
            outs, outs_warp = (
                self.net(img.to(self.device)),
                self.net(img_w.to(self.device)),
            )
        else:
            with torch.no_grad():
                outs, outs_warp = (
                    self.net(img.to(self.device)),
                    self.net(img_w.to(self.device)),
                )
        semi, coarse_desc = outs['semi'], outs['desc']
        semi_warp, coarse_desc_warp = outs_warp['semi'], outs_warp['desc']

        thd = 0.1

        desc = self.interpolate_to_dense(coarse_desc)
        desc_w = self.interpolate_to_dense(coarse_desc_warp)
        
        hms = thd_img(flattenDetection(semi), thd=thd)
        hms_w = thd_img(flattenDetection(semi_warp), thd=thd)
        loss = 0
        for b in range(self.batch_size):
            rs, cs = torch.where(hms[b][0] > 0)
            kpts = torch.stack((cs, rs), dim=1)
            rs_w, cs_w = torch.where(hms_w[b][0] > 0)
            kpts_w = torch.stack((cs_w, rs_w), dim=1)

            dbs= sample['kpts2D'][b].to(self.device)
            dbs_w= sample['kpts2D_w'][b].to(self.device)
            dbs, dbs_w = (dbs*0.5).int().float(), (dbs_w*0.5).int().float()

            output_kpts_idx = []
            output_3Dcoor = []
            dbslist = dbs.detach().cpu().tolist()
            dbslist_w = dbs_w.detach().cpu().tolist()
            for i, pt in enumerate(kpts.detach().cpu().tolist()):
                pt = self.pt_in_list(pt=pt, dblist=dbslist)
                if pt:
                    output_kpts_idx.append(i)            # output_kpts_idx: [2, 3, 5, 6, 7, 9, ...] db에 있는 kpt 인덱스
                    output_3Dcoor.append(sample['kpts3D'][b][dbslist.index(pt)])   # [2번의3D좌표, 3번의 3D좌표, 5번의 3D좌표, ...]

            output_kpts_idx_w = []
            output_3Dcoor_w = []
            for i, pt_w in enumerate(kpts_w.detach().cpu().tolist()):
                pt_w = self.pt_in_list(pt=pt_w, dblist=dbslist_w)
                if pt_w:
                    output_kpts_idx_w.append(i)
                    output_3Dcoor_w.append(sample['kpts3D_w'][b][dbslist_w.index(pt_w)])

            matched_idx, matched_idx_w = [], []
            for i1, k1 in enumerate(output_3Dcoor):
                for i2, k2 in enumerate(output_3Dcoor_w):
                    if torch.sum(torch.abs(k1-k2)) < 5.0e-2:
            #             print(k1, k2)
                        matched_idx.append(i1)
                        matched_idx_w.append(i2)
            matched_kpts_idx = torch.Tensor(output_kpts_idx)[matched_idx]
            matched_kpts_idx_w = torch.Tensor(output_kpts_idx_w)[matched_idx_w]
            ############################################################################3

            loss_desc = self.descriptor_loss(
                desc[b],
                desc_w[b],
                kpts, matched_kpts_idx, 
                kpts_w, matched_kpts_idx_w
            )
            if b == 0:
                if self.figname < 20:
                    self.visualize_kpts(
                        img, 
                        img_w, 
                        kpts[matched_kpts_idx.int()], 
                        kpts_w[matched_kpts_idx_w.int()], 
                        n=1000)
                    self.visualize = False
                    self.figname += 1
            loss += loss_desc

        self.loss = loss
        if train:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def pt_in_list(self, pt, dblist):
        for xvar in [-1, 0, 1]:
            for yvar in [-1, 0, 1]:
                if [pt[0]+xvar, pt[1]+yvar] in dblist:
                    return [pt[0]+xvar, pt[1]+yvar]
        return False

    def saveModel(self):
        """
        # save checkpoint for resuming training
        :return:
        """
        model_state_dict = self.net.state_dict()
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.n_iter,
        )
        pass

    def visualize_kpts(self, imname1, imname2, kpts1, kpts2, n=3):
        if type(imname1) == str:
            im1, im2 = np.array(Image.open(imname1)), np.array(Image.open(imname2))
        else:
            if imname1.shape[1] == 3:
                imname1 = imname1[0].transpose(0,1).transpose(1,2)
                imname2 = imname2[0].transpose(0,1).transpose(1,2)
            elif imname.shape[0] == 3:
                imname1 = imname1.transpose(0,1).transpose(1,2)
                imname2 = imname2.transpose(0,1).transpose(1,2)
            im1, im2 = imname1.detach().cpu().numpy(), imname2.detach().cpu().numpy()
        R = 2
        fig, ax = plt.subplots(1, 2) 
        
        ax[0].imshow(im1)
        i = 0
        for x, y in kpts1:
            ax[0].add_patch(plt.Circle((x, y), R, color='r'))
            ax[0].text(x, y, str(i))
            i += 1
            if i>=n:
                break
                
        ax[0].text(0, 100, '(0, 100)')
        ax[0].text(0, 0, '(0, 0)')
        i=0
        ax[1].imshow(im2)
        for x, y in kpts2:
            ax[1].add_patch(plt.Circle((x, y), R, color='r'))
            ax[1].text(x, y, str(i))
            i += 1
            if i>=n:
                break
        plt.savefig(os.path.join(self.save_path, f'figures/{self.figname}.png'))

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    ######## static methods ########
    @staticmethod
    def interpolate_to_dense(coarse_desc, cell_size=8):
        dense_desc = nn.functional.interpolate(
            coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear"
        )
        # norm the descriptor
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc

        dense_desc = norm_desc(dense_desc)
        return dense_desc


if __name__ == "__main__":
    # load config
    filename = "configs/superpoint_coco_test.yaml"
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.load(f)

    from utils.loader import dataLoader as dataLoader

    # data = dataLoader(config, dataset='hpatches')
    task = config["data"]["dataset"]

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data["train_loader"], data["val_loader"]

    # model_fe = Train_model_frontend(config)
    # print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_frontend(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    # epoch += 1
    try:
        model_fe.train()
    # catch exception
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
    # is_best = True
