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


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
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
            
        :param save_path:
        :param device:
        :param verbose:
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
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)

        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("batch_size: ", self.config["model"]["batch_size"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)
        pass

    def dataParallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
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
        :return:
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
    def writer(self):
        """
        # writer for tensorboard
        :return:
        """
        # print("get writer")
        return self._writer

    @writer.setter
    def writer(self, writer):
        print("set writer")
        self._writer = writer

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
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
        running_losses = []
        epoch = 0
        # Train one epoch
        while self.n_iter < self.max_iter:
            print("epoch: ", epoch)
            epoch += 1
            self.visualize = True
            for i, sample_train in tqdm(enumerate(self.train_loader)):
                # train one sample
                # loss_out = self.train_val_sample(sample_train, self.n_iter, True)
                loss_out = 0
                self.n_iter += 1
                running_losses.append(loss_out)
                # run validation
                if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                    logging.info("====== Validating...")
                    for j, sample_val in enumerate(self.val_loader):
                        self.train_val_sample(sample_val, self.n_iter + j, False)
                        if j > self.config.get("validation_size", 3):
                            break
                if self.n_iter % 100 == 0:
                    iterlog = f"iter {self.n_iter}, loss: {self.loss}"
                    print(iterlog)
                    self.write_log(iterlog)
                # save model
                if self.n_iter % self.config["save_interval"] == 0:
                    self.saveModel()
                # ending condition
                if self.n_iter > self.max_iter:
                    # end training
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
        # if self.config['data']['gaussian_label']['enable']:
        #     loss = loss_func_BCE(nn.functional.softmax(semi, dim=1), labels3D_in_loss)
        #     loss = (loss.sum(dim=1) * mask_3D_flattened).sum()
        # else:
        loss = loss_func(semi, labels3D_in_loss)
        loss = (loss * mask_3D_flattened).sum()
        loss = loss / (mask_3D_flattened.sum() + 1e-10)
        return loss

    def train_val_sample(self, sample, n_iter=0, train=False):
        """
        # deprecated: default train_val_sample
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """
        ## get the inputs
        img, img_w = sample['image'].to(self.device), sample['warped_image'].to(self.device)

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
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

    def add_single_image_to_tb(self, task, img_tensor, n_iter, name="img"):
        """
        # add image to tensorboard for visualization
        :param task:
        :param img_tensor:
        :param n_iter:
        :param name:
        :return:
        """
        if img_tensor.dim() == 4:
            for i in range(min(img_tensor.shape[0], 5)):
                self.writer.add_image(
                    task + "-" + name + "/%d" % i, img_tensor[i, :, :, :], n_iter
                )
        else:
            self.writer.add_image(task + "-" + name, img_tensor[:, :, :], n_iter)

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

    # tensorboard
    def addImg2tensorboard(
        self,
        img,
        labels_2D,
        semi,
        img_warp=None,
        labels_warp_2D=None,
        mask_warp_2D=None,
        semi_warp=None,
        mask_3D_flattened=None,
        task="training",
    ):
        """
        # deprecated: add images to tensorboard
        :param img:
        :param labels_2D:
        :param semi:
        :param img_warp:
        :param labels_warp_2D:
        :param mask_warp_2D:
        :param semi_warp:
        :param mask_3D_flattened:
        :param task:
        :return:
        """
        # print("add images to tensorboard")

        n_iter = self.n_iter
        semi_flat = flattenDetection(semi[0, :, :, :])
        semi_warp_flat = flattenDetection(semi_warp[0, :, :, :])

        thd = self.config["model"]["detection_threshold"]
        semi_thd = thd_img(semi_flat, thd=thd)
        semi_warp_thd = thd_img(semi_warp_flat, thd=thd)

        result_overlap = img_overlap(
            toNumpy(labels_2D[0, :, :, :]), toNumpy(semi_thd), toNumpy(img[0, :, :, :])
        )

        self.writer.add_image(
            task + "-detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_0.png"
        )  # rgb to bgr * 255

        result_overlap = img_overlap(
            toNumpy(labels_warp_2D[0, :, :, :]),
            toNumpy(semi_warp_thd),
            toNumpy(img_warp[0, :, :, :]),
        )
        self.writer.add_image(
            task + "-warp_detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_1.png"
        )  # rgb to bgr * 255

        mask_overlap = img_overlap(
            toNumpy(1 - mask_warp_2D[0, :, :, :]) / 2,
            np.zeros_like(toNumpy(img_warp[0, :, :, :])),
            toNumpy(img_warp[0, :, :, :]),
        )

        # writer.add_image(task + '_mask_valid_first_layer', mask_warp[0, :, :, :], n_iter)
        # writer.add_image(task + '_mask_valid_last_layer', mask_warp[-1, :, :, :], n_iter)
        ##### print to check
        # print("mask_2D shape: ", mask_warp_2D.shape)
        # print("mask_3D_flattened shape: ", mask_3D_flattened.shape)
        for i in range(self.batch_size):
            if i < 5:
                self.writer.add_image(
                    task + "-mask_warp_origin", mask_warp_2D[i, :, :, :], n_iter
                )
                self.writer.add_image(
                    task + "-mask_warp_3D_flattened", mask_3D_flattened[i, :, :], n_iter
                )
        # self.writer.add_image(task + '-mask_warp_origin-1', mask_warp_2D[1, :, :, :], n_iter)
        # self.writer.add_image(task + '-mask_warp_3D_flattened-1', mask_3D_flattened[1, :, :], n_iter)
        self.writer.add_image(task + "-mask_warp_overlay", mask_overlap, n_iter)

    def tb_scalar_dict(self, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            self.writer.add_scalar(task + "-" + element, losses[element], self.n_iter)
            # print (task, '-', element, ": ", losses[element].item())

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        """
        # add image dictionary to tensorboard
        :param task:
            str (train, val)
        :param tb_imgs:
        :param max_img:
            int - number of images
        :return:
        """
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                # print(f"element: {element}")
                self.writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    self.n_iter,
                )


    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(
                task + "-" + element, tb_dict[element], self.n_iter
            )
        pass

    def printLosses(self, losses, task="training"):
        """
        # print loss for tracking training
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            # print ('add to tb: ', element)
            print(task, "-", element, ": ", losses[element].item())

    def add2tensorboard_nms(self, img, labels_2D, semi, task="training", batch_size=1):
        """
        # deprecated:
        :param img:
        :param labels_2D:
        :param semi:
        :param task:
        :param batch_size:
        :return:
        """
        from utils.utils import getPtsFromHeatmap
        from utils.utils import box_nms

        boxNms = False
        n_iter = self.n_iter

        nms_dist = self.config["model"]["nms"]
        conf_thresh = self.config["model"]["detection_threshold"]
        # print("nms_dist: ", nms_dist)
        precision_recall_list = []
        precision_recall_boxnms_list = []
        for idx in range(batch_size):
            semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
            semi_flat = toNumpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, 0)
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
            semi_thd_nms_sample = np.zeros_like(semi_thd)
            semi_thd_nms_sample[
                pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
            ] = 1

            label_sample = torch.squeeze(labels_2D[idx, :, :, :])
            # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
            # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
            # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
            label_sample_nms_sample = label_sample

            if idx < 5:
                result_overlap = img_overlap(
                    np.expand_dims(label_sample_nms_sample, 0),
                    np.expand_dims(semi_thd_nms_sample, 0),
                    toNumpy(img[idx, :, :, :]),
                )
                self.writer.add_image(
                    task + "-detector_output_thd_overlay-NMS" + "/%d" % idx,
                    result_overlap,
                    n_iter,
                )
            assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
            precision_recall = precisionRecall_torch(
                torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample
            )
            precision_recall_list.append(precision_recall)

            if boxNms:
                semi_flat_tensor_nms = box_nms(
                    semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh
                ).cpu()
                semi_flat_tensor_nms = (semi_flat_tensor_nms >= conf_thresh).float()

                if idx < 5:
                    result_overlap = img_overlap(
                        np.expand_dims(label_sample_nms_sample, 0),
                        semi_flat_tensor_nms.numpy()[np.newaxis, :, :],
                        toNumpy(img[idx, :, :, :]),
                    )
                    self.writer.add_image(
                        task + "-detector_output_thd_overlay-boxNMS" + "/%d" % idx,
                        result_overlap,
                        n_iter,
                    )
                precision_recall_boxnms = precisionRecall_torch(
                    semi_flat_tensor_nms, label_sample_nms_sample
                )
                precision_recall_boxnms_list.append(precision_recall_boxnms)

        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        self.writer.add_scalar(task + "-precision_nms", precision, n_iter)
        self.writer.add_scalar(task + "-recall_nms", recall, n_iter)
        print(
            "-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f"
            % (task, n_iter, precision, recall)
        )
        if boxNms:
            precision = np.mean(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            recall = np.mean(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            self.writer.add_scalar(task + "-precision_boxnms", precision, n_iter)
            self.writer.add_scalar(task + "-recall_boxnms", recall, n_iter)
            print(
                "-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f"
                % (task, n_iter, precision, recall)
            )

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    ######## static methods ########
    @staticmethod
    def input_to_imgDict(sample, tb_images_dict):
        # for e in list(sample):
        #     print("sample[e]", sample[e].shape)
        #     if (sample[e]).dim() == 4:
        #         tb_images_dict[e] = sample[e]
        for e in list(sample):
            element = sample[e]
            if type(element) is torch.Tensor:
                if element.dim() == 4:
                    tb_images_dict[e] = element
                # print("shape of ", i, " ", element.shape)
        return tb_images_dict

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
